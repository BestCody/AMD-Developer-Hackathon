/** Forgot-password gate — mirrors LoginScreen's premium card treatment. Aperture addition. */
function ForgotPasswordScreen({ onBack }) {
  const [email, setEmail] = React.useState("");
  const [focus, setFocus] = React.useState(false);
  const [sent, setSent] = React.useState(false);

  const fieldStyle = {
    width: "100%",
    fontFamily: "var(--font-text)",
    fontSize: "var(--text-body-size)",
    color: "#000",
    background: "var(--surface-canvas)",
    border: `1px solid ${focus ? "var(--accent-primary)" : "var(--border-hairline)"}`,
    boxShadow: focus ? "0 0 0 3px var(--blue-50)" : "none",
    borderRadius: "var(--radius-sm)",
    height: 48,
    padding: "0 16px",
    outline: "none",
    boxSizing: "border-box",
    transition: "border-color var(--duration-fast) ease, box-shadow var(--duration-fast) ease",
  };

  return (
    <div
      style={{
        height: "100%",
        width: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "radial-gradient(circle at 50% 0%, var(--gray-100) 0%, var(--surface-parchment) 55%)",
      }}
    >
      <div
        style={{
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
        }}
      >
        <img src="../../assets/logo/aperture-mark.png" alt="Aperture" style={{ width: 52, height: 52, borderRadius: 13, boxShadow: "0 8px 20px -6px rgba(0,0,0,0.25)" }} />

        <div style={{ textAlign: "center" }}>
          <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-display-md-size)", fontWeight: 600, letterSpacing: "var(--text-display-md-tracking)", color: "#000" }}>
            Reset your password
          </div>
          <div style={{ fontSize: "var(--text-caption-size)", color: "#000", marginTop: 6, maxWidth: 360 }}>
            {sent ? "Check your inbox for a reset link." : "Enter your email and we'll send you a reset link."}
          </div>
        </div>

        {!sent && (
          <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 14 }}>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onFocus={() => setFocus(true)}
              onBlur={() => setFocus(false)}
              placeholder="Email"
              style={fieldStyle}
            />
          </div>
        )}

        {!sent && (
          <button
            onClick={() => setSent(true)}
            style={{
              width: "100%",
              background: "#000",
              color: "#fff",
              border: "none",
              borderRadius: "var(--radius-pill)",
              height: 48,
              fontFamily: "var(--font-text)",
              fontSize: "var(--text-body-size)",
              fontWeight: 500,
              cursor: "pointer",
              boxShadow: "0 8px 20px -6px rgba(0,0,0,0.35)",
              transition: "transform var(--duration-press) var(--ease-standard)",
            }}
            onMouseDown={(e) => (e.currentTarget.style.transform = "scale(var(--scale-press))")}
            onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
            onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
          >
            Send reset link
          </button>
        )}

        <button
          onClick={onBack}
          style={{
            border: "none", background: "transparent", cursor: "pointer",
            fontSize: "var(--text-caption-size)", color: "#000", display: "flex", alignItems: "center", gap: 6,
          }}
        >
          <i data-lucide="chevron-left" style={{ width: 14, height: 14 }}></i>
          Back to log in
        </button>
      </div>
    </div>
  );
}

window.ConsoleForgotPasswordScreen = ForgotPasswordScreen;
