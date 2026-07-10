/* AuthScreens.jsx -- login + signup, backed by /api/auth/*.
 *
 * Ported from the console design kit. Two deliberate changes from the
 * design prototype:
 *
 *   1. The prototype gated on `localStorage.getItem("aperture_logged_in")`,
 *      which any visitor could set from the devtools console. Auth now lives
 *      in a signed, HttpOnly session cookie the page cannot read or forge.
 *
 *   2. The prototype's "Forgot password?" link is gone. A real reset flow
 *      needs to deliver a single-use token out of band (email), and this
 *      project has no mail transport. A link that goes nowhere is worse than
 *      no link. See auth.py's module docstring.
 */

const AUTH_CARD = {
  width: 520,
  background: "var(--surface-canvas)",
  borderRadius: "var(--radius-lg)",
  boxShadow: "0 1px 2px rgba(0,0,0,0.04), 0 24px 60px -12px rgba(0,0,0,0.18)",
  border: "1px solid var(--border-hairline)",
  padding: "48px 40px 36px",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: 28,
};

const AUTH_SHELL = {
  height: "100%",
  width: "100%",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: "radial-gradient(circle at 50% 0%, var(--gray-100) 0%, var(--surface-parchment) 55%)",
};

function fieldStyle(focused) {
  return {
    width: "100%",
    fontFamily: "var(--font-text)",
    fontSize: "var(--text-body-size)",
    color: "var(--text-ink)",
    background: "var(--surface-canvas)",
    border: `1px solid ${focused ? "var(--accent-primary)" : "var(--border-hairline)"}`,
    boxShadow: focused ? "0 0 0 3px var(--blue-50)" : "none",
    borderRadius: "var(--radius-sm)",
    height: 48,
    padding: "0 16px",
    outline: "none",
    boxSizing: "border-box",
    transition: "border-color var(--duration-fast) ease, box-shadow var(--duration-fast) ease",
  };
}

const SUBMIT_STYLE = {
  width: "100%",
  background: "var(--gray-1000)",
  color: "var(--white)",
  border: "none",
  borderRadius: "var(--radius-pill)",
  height: 48,
  fontFamily: "var(--font-text)",
  fontSize: "var(--text-body-size)",
  fontWeight: 600,
  cursor: "pointer",
  boxShadow: "0 8px 20px -6px rgba(0,0,0,0.35)",
  transition: "transform var(--duration-press) var(--ease-standard)",
};

function pressHandlers() {
  return {
    onMouseDown: (e) => (e.currentTarget.style.transform = "scale(var(--scale-press))"),
    onMouseUp: (e) => (e.currentTarget.style.transform = "scale(1)"),
    onMouseLeave: (e) => (e.currentTarget.style.transform = "scale(1)"),
  };
}

function ErrorNote({ children }) {
  if (!children) return null;
  return (
    <div
      role="alert"
      style={{
        width: "100%",
        background: "var(--status-error-bg)",
        color: "var(--status-error)",
        borderRadius: "var(--radius-sm)",
        padding: "10px 14px",
        fontSize: "var(--text-caption-size)",
      }}
    >
      {children}
    </div>
  );
}

function Brand({ title, subtitle }) {
  return (
    <>
      <img
        src="/static/ds/assets/logo/aperture-mark.png"
        alt=""
        style={{ width: 52, height: 52, borderRadius: 13, boxShadow: "0 8px 20px -6px rgba(0,0,0,0.25)" }}
      />
      <div style={{ textAlign: "center" }}>
        <div
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "var(--text-display-md-size)",
            fontWeight: 600,
            letterSpacing: "var(--text-display-md-tracking)",
            color: "var(--text-ink)",
          }}
        >
          {title}
        </div>
        <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)", marginTop: 6 }}>
          {subtitle}
        </div>
      </div>
    </>
  );
}

function LoginScreen({ onAuthed, onSignUp }) {
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [focus, setFocus] = React.useState(null);
  const [error, setError] = React.useState("");
  const [busy, setBusy] = React.useState(false);

  async function submit(e) {
    if (e) e.preventDefault();
    if (busy) return;
    setError("");
    setBusy(true);
    try {
      const { user } = await window.MonadLabsAPI.login(email, password);
      onAuthed(user);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  return (
    <div style={AUTH_SHELL}>
      <form style={AUTH_CARD} onSubmit={submit}>
        <Brand title="Welcome back" subtitle="One format, every modality." />
        <ErrorNote>{error}</ErrorNote>
        <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 14 }}>
          <input
            type="email" autoComplete="email" required value={email}
            onChange={(e) => setEmail(e.target.value)}
            onFocus={() => setFocus("email")} onBlur={() => setFocus(null)}
            placeholder="Email" style={fieldStyle(focus === "email")}
          />
          <input
            type="password" autoComplete="current-password" required value={password}
            onChange={(e) => setPassword(e.target.value)}
            onFocus={() => setFocus("password")} onBlur={() => setFocus(null)}
            placeholder="Password" style={fieldStyle(focus === "password")}
          />
        </div>
        <button type="submit" disabled={busy} style={{ ...SUBMIT_STYLE, opacity: busy ? 0.6 : 1 }} {...pressHandlers()}>
          {busy ? "Signing in…" : "Log in"}
        </button>
        <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)" }}>
          No account?{" "}
          <a href="#" onClick={(e) => { e.preventDefault(); onSignUp(); }} style={{ color: "var(--accent-primary)" }}>
            Sign up
          </a>
        </div>
      </form>
    </div>
  );
}

function SignUpScreen({ onAuthed, onBack }) {
  const [name, setName] = React.useState("");
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [focus, setFocus] = React.useState(null);
  const [error, setError] = React.useState("");
  const [busy, setBusy] = React.useState(false);

  async function submit(e) {
    if (e) e.preventDefault();
    if (busy) return;
    setError("");
    setBusy(true);
    try {
      const { user } = await window.MonadLabsAPI.signup(email, password, name);
      onAuthed(user);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  return (
    <div style={AUTH_SHELL}>
      <form style={AUTH_CARD} onSubmit={submit}>
        <Brand title="Create your account" subtitle="One format, every modality." />
        <ErrorNote>{error}</ErrorNote>
        <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 14 }}>
          <input
            type="text" autoComplete="name" value={name}
            onChange={(e) => setName(e.target.value)}
            onFocus={() => setFocus("name")} onBlur={() => setFocus(null)}
            placeholder="Full name" style={fieldStyle(focus === "name")}
          />
          <input
            type="email" autoComplete="email" required value={email}
            onChange={(e) => setEmail(e.target.value)}
            onFocus={() => setFocus("email")} onBlur={() => setFocus(null)}
            placeholder="Email" style={fieldStyle(focus === "email")}
          />
          <input
            type="password" autoComplete="new-password" required minLength={8} value={password}
            onChange={(e) => setPassword(e.target.value)}
            onFocus={() => setFocus("password")} onBlur={() => setFocus(null)}
            placeholder="Password (8+ characters)" style={fieldStyle(focus === "password")}
          />
        </div>
        <button type="submit" disabled={busy} style={{ ...SUBMIT_STYLE, opacity: busy ? 0.6 : 1 }} {...pressHandlers()}>
          {busy ? "Creating account…" : "Sign up"}
        </button>
        <button
          type="button" onClick={onBack}
          style={{ border: "none", background: "transparent", cursor: "pointer", fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)" }}
        >
          Back to log in
        </button>
      </form>
    </div>
  );
}

window.ConsoleLoginScreen = LoginScreen;
window.ConsoleSignUpScreen = SignUpScreen;
