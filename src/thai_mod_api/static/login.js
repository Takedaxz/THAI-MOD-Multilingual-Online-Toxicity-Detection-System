const loginForm = document.getElementById("loginForm");
const usernameInput = document.getElementById("usernameInput");
const passwordInput = document.getElementById("passwordInput");
const loginButton = document.getElementById("loginButton");
const loginError = document.getElementById("loginError");

function getNextPath() {
  const params = new URLSearchParams(window.location.search);
  const next = params.get("next");
  if (!next || !next.startsWith("/") || next.startsWith("//")) {
    return "/admin";
  }
  return next;
}

function showError(message) {
  loginError.textContent = message;
  loginError.classList.remove("hidden");
}

async function handleSubmit(event) {
  event.preventDefault();
  loginError.classList.add("hidden");

  loginButton.disabled = true;
  loginButton.textContent = "Signing in...";

  try {
    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        username: usernameInput.value.trim(),
        password: passwordInput.value,
        next_path: getNextPath(),
      }),
    });

    if (!response.ok) {
      if (response.status === 401) {
        throw new Error("Invalid username or password");
      }
      throw new Error(`Login failed (${response.status})`);
    }

    const payload = await response.json();
    const destination = payload.next_path || "/admin";
    window.location.assign(destination);
  } catch (error) {
    showError(String(error.message || error));
  } finally {
    loginButton.disabled = false;
    loginButton.textContent = "Sign in";
  }
}

loginForm.addEventListener("submit", handleSubmit);
