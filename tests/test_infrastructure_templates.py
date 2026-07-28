"""Regression checks for the AWS instance bootstrap and network."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
HOST_NETWORK_SERVICE_COUNT = 2


def test_application_network_supports_ipv6_database_egress():
    terraform = (ROOT / "infra/terraform/application/main.tf").read_text(encoding="utf-8")

    assert "assign_generated_ipv6_cidr_block = true" in terraform
    assert 'ipv6_cidr_block = "::/0"' in terraform
    assert 'ipv6_cidr_blocks = ["::/0"]' in terraform
    assert "ipv6_address_count          = 1" in terraform


def test_instance_bootstrap_uses_available_al2023_packages_and_compose_env():
    template = (
        ROOT / "infra/terraform/application/user_data.sh.tftpl"
    ).read_text(encoding="utf-8")

    assert "dnf install -y docker jq\n" in template
    assert "awscli2" not in template
    assert "docker compose --env-file /opt/quoptuna/compose.env" in template


def test_deployment_waits_for_cloud_init():
    operation = (ROOT / "infra/scripts/_operation.sh").read_text(encoding="utf-8")

    assert 'run_ssm "$instance_id" "cloud-init status --wait"' in operation


def test_shell_cleanup_traps_are_scoped_to_subshells():
    scripts = [
        (ROOT / "infra/scripts/_operation.sh").read_text(encoding="utf-8"),
        (ROOT / "infra/scripts/common.sh").read_text(encoding="utf-8"),
    ]

    assert all("trap " + "'rm " not in script or " RETURN" not in script for script in scripts)


def test_update_ignores_stale_instance_output_after_partial_apply():
    operation = (ROOT / "infra/scripts/_operation.sh").read_text(encoding="utf-8")

    assert (
        'terraform -chdir="$APPLICATION_DIR" state show aws_instance.app'
        in operation
    )


def test_runtime_uses_host_network_for_supabase_ipv6():
    compose = (ROOT / "infra/runtime/docker-compose.prod.yml").read_text(
        encoding="utf-8"
    )
    caddy = (ROOT / "infra/runtime/Caddyfile").read_text(encoding="utf-8")

    assert compose.count("network_mode: host") == HOST_NETWORK_SERVICE_COUNT
    assert "reverse_proxy 127.0.0.1:8000" in caddy
    assert "reverse_proxy app:8000" not in caddy
