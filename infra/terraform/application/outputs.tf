output "instance_id" {
  value = aws_instance.app.id
}

output "public_ip" {
  value = aws_instance.app.public_ip
}

output "application_url" {
  value = "https://${var.domain_name}"
}

output "domain_name" {
  value = var.domain_name
}

output "artifact_bucket" {
  value = var.artifact_bucket
}

output "ecr_repository_url" {
  value = var.ecr_repository_url
}

output "ecr_repository_name" {
  value = var.ecr_repository_name
}

output "runtime_secret_name" {
  value = var.runtime_secret_name
}
