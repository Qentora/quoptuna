# QuOptuna infrastructure

The Textual operator console runs the executable scripts in `scripts/`; it does
not invoke Terraform directly. Add environment-specific Terraform configuration
under `environments/dev` and `environments/production`, then replace the script
stubs with the approved create, deploy, update, pause, resume, status, and destroy
operations.

Start the console with:

```bash
uv run quoptuna infra --environment dev --env-file .env
```

Do not commit `.env`, Terraform state, Supabase URLs, AWS credentials, or other
secrets. Persistent application data belongs in Supabase/PostgreSQL and result
artifacts belong in S3; pausing compute must not delete either.
