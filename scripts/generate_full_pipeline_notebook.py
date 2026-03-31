from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "toxicity_detection.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


cells = [
    md(
        """
        # THAI-MOD Full Pipeline

        This notebook combines the workflows from `model.ipynb` and `thai-bert.ipynb` into one end-to-end pipeline for the THAI-MOD system.

        The structure and assumptions here are based on:
        - `docs/proposal.txt`
        - `docs/progress/progress1.txt`
        - `docs/progress/progress2.txt`

        ## What this notebook covers
        - Dataset aggregation from all 8 documented datasets
        - Text preprocessing for Thai + English comments
        - Stratified train/test split
        - Traditional ML baselines with TF-IDF features
        - Transformer model comparison
        - Confusion matrix for each model
        - Final inference helper aligned with the system pipeline
        """
    ),
    md(
        """
        ## System Pipeline From The Docs

        The notebook follows the project docs:
        1. Aggregate multilingual toxicity datasets and deduplicate by text.
        2. Clean text by removing URLs, normalizing emojis, and lowercasing English characters.
        3. Map labels into binary classes: `toxic = 1`, `non-toxic = 0`.
        4. Split data with stratification to preserve class balance.
        5. Train traditional TF-IDF baselines and transformer-based models.
        6. Evaluate with recall-oriented metrics and confusion matrices.
        7. Provide a final decision-support inference helper for moderators.

        The docs emphasize recall on the toxic class because false negatives are more harmful than false positives in moderation.
        """
    ),
    code(
        """
        import gc
        import re
        import string
        import warnings
        from pathlib import Path

        import emoji
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import torch
        import torch.nn.functional as F
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import RandomOverSampler
        from pythainlp.tokenize import word_tokenize
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            fbeta_score,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            CamembertTokenizer,
            get_linear_schedule_with_warmup,
        )
        from xgboost import XGBClassifier

        warnings.filterwarnings("ignore")
        pd.set_option("display.max_colwidth", None)
        sns.set_theme(style="whitegrid")
        """
    ),
    code(
        """
        ROOT = Path.cwd()
        DATASET_FILES = [ROOT / "datasets" / f"dataset{i}.csv" for i in range(1, 9)]
        MODEL_CACHE_DIR = ROOT / "models"
        BASELINE_CACHE_DIR = MODEL_CACHE_DIR / "baselines"
        TRANSFORMER_CACHE_DIR = MODEL_CACHE_DIR / "transformers"
        BASELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        TRANSFORMER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        RANDOM_STATE = 42
        TEST_SIZE = 0.2

        MAX_FEATURES = 10_000
        NGRAM_RANGE = (1, 2)
        CLASS_NAMES = ["non-toxic", "toxic"]
        TRANSFORMER_THRESHOLD = 0.4

        EPOCHS = 3
        LEARNING_RATE = 2e-5
        WARMUP_STEPS = 100

        TRANSFORMER_MODELS = {
            "WangchanBERTa": "airesearch/wangchanberta-base-att-spm-uncased",
            "XLM-RoBERTa": "xlm-roberta-base",
            "PhayaThaiBERT": "clicknext/phayathaibert",
        }

        ROOT, DATASET_FILES[:2], TRANSFORMER_MODELS
        """
    ),
    code(
        """
        def clean_text(text):
            if pd.isna(text):
                return text

            text = str(text)
            text = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\\\(\\\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                text,
            )
            text = re.sub(
                r"www\\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\\\(\\\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                text,
            )
            text = emoji.demojize(text, language="en")
            return text.strip()


        def lowercase_english(text):
            if pd.isna(text):
                return text

            result = []
            for char in str(text):
                result.append(char.lower() if ord(char) < 128 else char)
            return "".join(result)


        def calculate_thai_length(text):
            if pd.isna(text):
                return 0
            return len(word_tokenize(str(text), engine="newmm"))


        def prepare_dataset(dataset_file):
            df = pd.read_csv(dataset_file).copy()
            df = df.dropna(subset=["category", "texts"])
            df["texts"] = df["texts"].apply(clean_text).apply(lowercase_english)
            df = df[df["texts"].str.strip() != ""].copy()
            df["category"] = df["category"].replace({"pos": "neu"})
            df["category"] = df["category"].map({"neg": 1, "neu": 0})
            df = df.dropna(subset=["category"]).copy()
            df["category"] = df["category"].astype(int)
            df["length"] = df["texts"].apply(calculate_thai_length)
            df["source"] = Path(dataset_file).name
            return df[["texts", "category", "length", "source"]]


        def load_full_dataset(dataset_files):
            prepared = [prepare_dataset(path) for path in dataset_files]
            combined = pd.concat(prepared, ignore_index=True)
            before_dedup = len(combined)
            combined = combined.drop_duplicates(subset=["texts"], keep="first").reset_index(drop=True)
            print(f"Rows before deduplication: {before_dedup:,}")
            print(f"Rows after deduplication:  {len(combined):,}")
            print(f"Label distribution: {combined['category'].value_counts(normalize=True).round(4).to_dict()}")
            return combined


        combined_df = load_full_dataset(DATASET_FILES)
        combined_df.head()
        """
    ),
    code(
        """
        summary_df = pd.DataFrame(
            {
                "rows": [len(combined_df)],
                "toxic_count": [int(combined_df["category"].sum())],
                "non_toxic_count": [int((combined_df["category"] == 0).sum())],
                "avg_length": [round(combined_df["length"].mean(), 2)],
                "median_length": [round(combined_df["length"].median(), 2)],
            }
        )
        summary_df
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        class_counts = combined_df["category"].map({0: "non-toxic", 1: "toxic"}).value_counts()
        sns.barplot(x=class_counts.index, y=class_counts.values, ax=axes[0], palette="Set2")
        axes[0].set_title("Class Distribution")
        axes[0].set_xlabel("Label")
        axes[0].set_ylabel("Count")

        source_counts = combined_df["source"].value_counts().sort_values(ascending=False)
        sns.barplot(x=source_counts.values, y=source_counts.index, ax=axes[1], palette="crest")
        axes[1].set_title("Samples Per Dataset")
        axes[1].set_xlabel("Count")
        axes[1].set_ylabel("Source")

        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        X = combined_df["texts"]
        y = combined_df["category"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        print("Train shape:", X_train.shape, y_train.shape)
        print("Test shape: ", X_test.shape, y_test.shape)
        print("Train class distribution:", y_train.value_counts(normalize=True).round(4).to_dict())
        print("Test class distribution: ", y_test.value_counts(normalize=True).round(4).to_dict())
        """
    ),
    md(
        """
        ## Traditional ML Models

        This section follows the baseline comparison from `model.ipynb` and the docs:
        - Multinomial Naive Bayes
        - Linear SVC
        - Logistic Regression
        - Random Forest
        - XGBoost

        TF-IDF features with Thai tokenization are used as the classical feature pipeline.
        """
    ),
    code(
        """
        def get_safe_name(name):
            return name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


        def process_text(text):
            nopunc = "".join(char for char in str(text) if char not in string.punctuation)
            return [token for token in word_tokenize(nopunc, engine="newmm") if token and not token.isspace()]


        def build_classical_models():
            return {
                "Naive Bayes": Pipeline(
                    [
                        ("vect", CountVectorizer(tokenizer=process_text, ngram_range=NGRAM_RANGE, token_pattern=None, min_df=2, max_features=MAX_FEATURES)),
                        ("tfidf", TfidfTransformer()),
                        ("clf", MultinomialNB(alpha=0.001)),
                    ]
                ),
                "Linear SVC": Pipeline(
                    [
                        ("vect", CountVectorizer(tokenizer=process_text, ngram_range=NGRAM_RANGE, token_pattern=None, min_df=2, max_features=MAX_FEATURES)),
                        ("tfidf", TfidfTransformer()),
                        ("clf", LinearSVC()),
                    ]
                ),
                "Logistic Regression": Pipeline(
                    [
                        ("vect", CountVectorizer(tokenizer=process_text, ngram_range=NGRAM_RANGE, token_pattern=None)),
                        ("tfidf", TfidfTransformer()),
                        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
                    ]
                ),
                "Random Forest": Pipeline(
                    [
                        ("vect", CountVectorizer(tokenizer=process_text, ngram_range=NGRAM_RANGE, token_pattern=None)),
                        ("tfidf", TfidfTransformer()),
                        ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
                    ]
                ),
                "XGBoost": Pipeline(
                    [
                        ("vect", CountVectorizer(tokenizer=process_text, ngram_range=NGRAM_RANGE, token_pattern=None)),
                        ("tfidf", TfidfTransformer()),
                        ("clf", XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss")),
                    ]
                ),
            }
        """
    ),
    code(
        """
        def compute_metrics(y_true, y_pred):
            return {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1-Score": f1_score(y_true, y_pred, zero_division=0),
                "F2-Score": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            }


        def plot_confusion_matrices(results, title, cols=2):
            rows = int(np.ceil(len(results) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            axes = np.array(axes).reshape(-1)

            for ax, result in zip(axes, results):
                sns.heatmap(
                    result["confusion_matrix"],
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    ax=ax,
                    xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES,
                )
                ax.set_title(result["Model"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")

            for ax in axes[len(results):]:
                ax.axis("off")

            fig.suptitle(title, fontsize=16, y=1.02)
            plt.tight_layout()
            plt.show()


        def evaluate_classical_models(models, X_train, X_test, y_train, y_test):
            results = []
            fitted_models = {}

            for model_name, model in models.items():
                print(f"Training {model_name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred)
                metrics["Model"] = model_name
                metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred)
                results.append(metrics)
                fitted_models[model_name] = model

                print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

            return results, fitted_models
        """
    ),
    code(
        """
        def train_or_load_baseline_model(
            model_name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            force_retrain=False,
        ):
            safe_name = get_safe_name(model_name)
            model_path = BASELINE_CACHE_DIR / f"{safe_name}.joblib"
            meta_path = BASELINE_CACHE_DIR / f"{safe_name}_metrics.json"

            if model_path.exists() and meta_path.exists() and not force_retrain:
                print(f"Loading cached baseline: {model_name}")
                fitted_model = joblib.load(model_path)

                with open(meta_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)

                metrics["confusion_matrix"] = np.array(metrics["confusion_matrix"])
                return metrics, fitted_model

            print(f"Training baseline: {model_name}")
            fitted_model = model.fit(X_train, y_train)
            y_pred = fitted_model.predict(X_test)

            metrics = compute_metrics(y_test, y_pred)
            metrics["Model"] = model_name
            metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred)

            joblib.dump(fitted_model, model_path)

            serializable_metrics = {
                "Model": metrics["Model"],
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-Score": metrics["F1-Score"],
                "F2-Score": metrics["F2-Score"],
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)

            return metrics, fitted_model
        """
    ),
    code(
        """
        classical_models = build_classical_models()
        classical_results = []
        fitted_classical_models = {}

        for model_name, model in classical_models.items():
            metrics, fitted_model = train_or_load_baseline_model(
                model_name,
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                force_retrain=False,
            )
            classical_results.append(metrics)
            fitted_classical_models[model_name] = fitted_model

        classical_results_df = (
            pd.DataFrame(classical_results)
            .drop(columns=["confusion_matrix"])
            .sort_values(by=["Recall", "F2-Score", "F1-Score"], ascending=False)
            .reset_index(drop=True)
        )
        classical_results_df
        """
    ),
    code(
        """
        plot_confusion_matrices(classical_results, "Confusion Matrices: Traditional ML Models", cols=2)
        """
    ),
    code(
        """
        metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score"]
        melted = classical_results_df.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")

        plt.figure(figsize=(14, 6))
        sns.barplot(data=melted, x="Metric", y="Score", hue="Model")
        plt.ylim(0, 1.0)
        plt.title("Traditional Model Comparison")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Recall-Oriented Traditional Variants

        The project docs mention three techniques to improve toxic recall:
        - `class_weight='balanced'`
        - oversampling
        - threshold tuning

        This section keeps those experiments in the combined notebook as optional recall-focused variants.
        """
    ),
    code(
        """
        svc_balanced = Pipeline(
            [
                ("vect", TfidfVectorizer(tokenizer=word_tokenize, ngram_range=NGRAM_RANGE, min_df=5)),
                ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )

        lr_balanced = Pipeline(
            [
                ("vect", TfidfVectorizer(tokenizer=word_tokenize, ngram_range=NGRAM_RANGE, min_df=5)),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
            ]
        )

        svc_oversampled = ImbPipeline(
            [
                ("vect", TfidfVectorizer(tokenizer=word_tokenize, ngram_range=NGRAM_RANGE, min_df=5)),
                ("over", RandomOverSampler(random_state=RANDOM_STATE)),
                ("clf", LinearSVC(random_state=RANDOM_STATE)),
            ]
        )

        recall_variant_models = {
            "Linear SVC (Balanced)": svc_balanced,
            "Logistic Regression (Balanced)": lr_balanced,
            "Linear SVC (Oversampled)": svc_oversampled,
        }

        recall_variant_results, recall_variant_fitted = evaluate_classical_models(
            recall_variant_models, X_train, X_test, y_train, y_test
        )

        threshold_probs = fitted_classical_models["Logistic Regression"].predict_proba(X_test)[:, 1]
        threshold_preds = (threshold_probs >= 0.4).astype(int)
        threshold_metrics = compute_metrics(y_test, threshold_preds)
        threshold_metrics["Model"] = "Logistic Regression (Threshold 0.4)"
        threshold_metrics["confusion_matrix"] = confusion_matrix(y_test, threshold_preds)
        recall_variant_results.append(threshold_metrics)

        recall_variant_df = (
            pd.DataFrame(recall_variant_results)
            .drop(columns=["confusion_matrix"])
            .sort_values(by=["Recall", "F2-Score", "F1-Score"], ascending=False)
            .reset_index(drop=True)
        )
        recall_variant_df
        """
    ),
    code(
        """
        plot_confusion_matrices(recall_variant_results, "Confusion Matrices: Recall-Oriented Variants", cols=2)
        """
    ),
    md(
        """
        ## Transformer Models

        This section combines the comparison workflow from `thai-bert.ipynb` with the recall-oriented evaluation described in the docs.

        Models:
        - WangchanBERTa
        - XLM-RoBERTa
        - PhayaThaiBERT

        Notes:
        - The docs identify WangchanBERTa as the selected model for the overall system.
        - A threshold of around `0.3-0.4` can improve toxic recall during moderation, so this notebook evaluates transformer outputs with `0.4` by default.
        """
    ),
    code(
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            MAX_LENGTH = 128
            BATCH_SIZE = 16
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            MAX_LENGTH = 96
            BATCH_SIZE = 16
        else:
            device = torch.device("cpu")
            MAX_LENGTH = 64
            BATCH_SIZE = 8

        print("Device:", device)
        print("MAX_LENGTH:", MAX_LENGTH)
        print("BATCH_SIZE:", BATCH_SIZE)
        """
    ),
    code(
        """
        class ToxicityDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = list(texts)
                self.labels = list(labels)
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    str(self.texts[idx]),
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                return {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
                }


        def get_label_weights(labels):
            labels = np.asarray(labels)
            counts = np.bincount(labels, minlength=2)
            weights = counts.sum() / (len(counts) * counts)
            return torch.tensor(weights, dtype=torch.float)


        class_weights = get_label_weights(y_train)
        class_weights
        """
    ),
    code(
        """
        def get_transformer_cache_paths(model_name):
            safe_name = get_safe_name(model_name)
            model_dir = TRANSFORMER_CACHE_DIR / safe_name
            meta_path = model_dir / "metadata.json"
            return model_dir, meta_path


        def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights):
            model.train()
            total_loss = 0.0
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

            for batch in tqdm(dataloader, desc="Training", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            return total_loss / max(len(dataloader), 1)


        def evaluate_transformer(model, dataloader, device, threshold=0.4):
            model.eval()
            total_loss = 0.0
            preds = []
            probs = []
            labels_all = []
            loss_fn = torch.nn.CrossEntropyLoss()

            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)
                    total_loss += loss.item()

                    toxic_probs = F.softmax(logits, dim=1)[:, 1]
                    batch_preds = (toxic_probs >= threshold).long()

                    probs.extend(toxic_probs.cpu().numpy())
                    preds.extend(batch_preds.cpu().numpy())
                    labels_all.extend(labels.cpu().numpy())

            return {
                "loss": total_loss / max(len(dataloader), 1),
                "predictions": np.array(preds),
                "probabilities": np.array(probs),
                "labels": np.array(labels_all),
            }


        def train_and_evaluate_transformer(
            model_name,
            model_id,
            X_train,
            y_train,
            X_test,
            y_test,
            threshold=0.4,
            force_retrain=False,
        ):
            model_dir, meta_path = get_transformer_cache_paths(model_name)

            if model_dir.exists() and meta_path.exists() and not force_retrain:
                print("=" * 90)
                print(f"Loading cached {model_name} from {model_dir}")
                print("=" * 90)

                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                if "wangchanberta" in model_id.lower():
                    tokenizer = CamembertTokenizer.from_pretrained(model_dir)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                )
                model.to(device)

                final_metrics = metadata["metrics"]
                final_metrics["confusion_matrix"] = np.array(final_metrics["confusion_matrix"])
                final_metrics["history"] = pd.DataFrame(metadata.get("history", []))

                trained_objects = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "threshold": metadata.get("threshold", threshold),
                    "model_dir": str(model_dir),
                    "loaded_from_cache": True,
                }

                return final_metrics, trained_objects

            print("=" * 90)
            print(f"Training {model_name}: {model_id}")
            print("=" * 90)

            if "wangchanberta" in model_id.lower():
                tokenizer = CamembertTokenizer.from_pretrained(model_id)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                num_labels=2,
                trust_remote_code=True,
            )
            model.to(device)

            train_dataset = ToxicityDataset(X_train, y_train, tokenizer, MAX_LENGTH)
            test_dataset = ToxicityDataset(X_test, y_test, tokenizer, MAX_LENGTH)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
            total_steps = len(train_loader) * EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=min(WARMUP_STEPS, total_steps // 10 if total_steps else 0),
                num_training_steps=total_steps,
            )

            best_state_dict = None
            best_f2 = -1.0
            history = []

            for epoch in range(EPOCHS):
                train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
                eval_output = evaluate_transformer(model, test_loader, device, threshold=threshold)
                epoch_f2 = fbeta_score(eval_output["labels"], eval_output["predictions"], beta=2, zero_division=0)

                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "eval_loss": eval_output["loss"],
                        "f2": epoch_f2,
                    }
                )

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"train_loss={train_loss:.4f} | eval_loss={eval_output['loss']:.4f} | f2={epoch_f2:.4f}"
                )

                if epoch_f2 > best_f2:
                    best_f2 = epoch_f2
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)
                model.to(device)

            final_eval = evaluate_transformer(model, test_loader, device, threshold=threshold)
            final_metrics = compute_metrics(final_eval["labels"], final_eval["predictions"])
            final_metrics["Model"] = model_name
            final_metrics["Threshold"] = threshold
            final_metrics["confusion_matrix"] = confusion_matrix(final_eval["labels"], final_eval["predictions"])
            final_metrics["history"] = pd.DataFrame(history)

            print(classification_report(final_eval["labels"], final_eval["predictions"], target_names=CLASS_NAMES, zero_division=0))

            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

            metadata = {
                "model_name": model_name,
                "model_id": model_id,
                "threshold": threshold,
                "metrics": {
                    "Model": final_metrics["Model"],
                    "Accuracy": final_metrics["Accuracy"],
                    "Precision": final_metrics["Precision"],
                    "Recall": final_metrics["Recall"],
                    "F1-Score": final_metrics["F1-Score"],
                    "F2-Score": final_metrics["F2-Score"],
                    "Threshold": final_metrics["Threshold"],
                    "confusion_matrix": final_metrics["confusion_matrix"].tolist(),
                },
                "history": history,
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            trained_objects = {
                "model": model,
                "tokenizer": tokenizer,
                "threshold": threshold,
                "model_dir": str(model_dir),
                "loaded_from_cache": False,
            }

            return final_metrics, trained_objects
        """
    ),
    code(
        """
        transformer_results = []
        trained_transformers = {}

        for model_name, model_id in TRANSFORMER_MODELS.items():
            result, trained_object = train_and_evaluate_transformer(
                model_name,
                model_id,
                X_train.tolist(),
                y_train.tolist(),
                X_test.tolist(),
                y_test.tolist(),
                threshold=TRANSFORMER_THRESHOLD,
                force_retrain=False,
            )
            transformer_results.append(result)
            trained_transformers[model_name] = trained_object

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

        transformer_results_df = (
            pd.DataFrame(transformer_results)
            .drop(columns=["confusion_matrix", "history"])
            .sort_values(by=["Recall", "F2-Score", "F1-Score"], ascending=False)
            .reset_index(drop=True)
        )
        transformer_results_df
        """
    ),
    code(
        """
        plot_confusion_matrices(transformer_results, "Confusion Matrices: Transformer Models", cols=2)
        """
    ),
    code(
        """
        transformer_metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score"]
        transformer_melted = transformer_results_df.melt(
            id_vars="Model",
            value_vars=transformer_metric_cols,
            var_name="Metric",
            value_name="Score",
        )

        plt.figure(figsize=(14, 6))
        sns.barplot(data=transformer_melted, x="Metric", y="Score", hue="Model")
        plt.ylim(0, 1.0)
        plt.title("Transformer Model Comparison")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        combined_results_df = pd.concat(
            [
                classical_results_df.assign(Group="Traditional"),
                transformer_results_df.assign(Group="Transformer"),
            ],
            ignore_index=True,
        ).sort_values(by=["Recall", "F2-Score", "F1-Score"], ascending=False)

        combined_results_df
        """
    ),
    md(
        """
        ## Final Inference Helper

        The docs position the system as a decision-support tool for moderators.

        The helper below uses the selected transformer model, returns:
        - predicted label
        - toxic score
        - confidence
        - moderation action recommendation

        Recommended moderation policy from the docs:
        - `HIDE` when toxic score is above threshold
        - `ALLOW` otherwise

        You can switch `SELECTED_MODEL_NAME` if you prefer another trained transformer after comparison.
        """
    ),
    code(
        """
        SELECTED_MODEL_NAME = "WangchanBERTa"
        selected_bundle = trained_transformers[SELECTED_MODEL_NAME]


        def preprocess_for_inference(text):
            return lowercase_english(clean_text(text))


        def predict_toxicity(text, bundle=selected_bundle):
            model = bundle["model"]
            tokenizer = bundle["tokenizer"]
            threshold = bundle["threshold"]

            processed = preprocess_for_inference(text)
            model.eval()

            encoded = tokenizer(
                processed,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.no_grad():
                logits = model(**encoded).logits
                toxic_score = F.softmax(logits, dim=1)[0, 1].item()

            predicted_label = int(toxic_score >= threshold)
            confidence = toxic_score if predicted_label == 1 else 1 - toxic_score

            return {
                "text": text,
                "processed_text": processed,
                "predicted_label": "toxic" if predicted_label == 1 else "non-toxic",
                "toxic_score": round(toxic_score, 4),
                "confidence": round(confidence, 4),
                "threshold": threshold,
                "action": "HIDE" if predicted_label == 1 else "ALLOW",
            }


        sample_texts = [
            "ข้อความนี้แย่มากและควรถูกลบ",
            "ขอบคุณมากครับ วันนี้ช่วยได้เยอะเลย",
            "this comment is extremely toxic and hateful",
        ]

        pd.DataFrame([predict_toxicity(text) for text in sample_texts])
        """
    ),
    md(
        """
        ## Suggested Conclusion

        After running the notebook end to end, use this section to summarize:
        - which traditional model performed best
        - which transformer performed best
        - whether WangchanBERTa remains the best deployment choice
        - which threshold gives the safest recall/precision trade-off for moderation

        If your final report follows the docs, prioritize:
        - recall for toxic comments
        - F2-score as a recall-supporting metric
        - confusion matrix analysis for false negatives and false positives
        """
    ),
]


notebook = nbf.v4.new_notebook()
notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "cedt",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11.13",
    },
}

OUTPUT.write_text(nbf.writes(notebook), encoding="utf-8")
print(f"Notebook written to {OUTPUT}")
