/**
 * Auth0 Post-Login Action for QuOptuna.
 *
 * Required Action secrets:
 *   QUOPTUNA_CLIENT_ID - the Auth0 application client ID
 *   ALLOWED_EMAILS     - comma-separated exact email addresses
 */
exports.onExecutePostLogin = async (event, api) => {
  if (event.client.client_id !== event.secrets.QUOPTUNA_CLIENT_ID) {
    return;
  }

  const allowed = new Set(
    (event.secrets.ALLOWED_EMAILS || "")
      .split(",")
      .map((email) => email.trim().toLowerCase())
      .filter(Boolean),
  );
  const email = (event.user.email || "").trim().toLowerCase();

  if (!event.user.email_verified) {
    api.access.deny("A verified email address is required.");
    return;
  }
  if (!email || !allowed.has(email)) {
    api.access.deny("This email address is not approved for QuOptuna.");
  }
};
