output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.id
}

output "artifact_bucket_arn" {
  value = aws_s3_bucket.artifacts.arn
}

output "ecr_repository_name" {
  value = aws_ecr_repository.app.name
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "ecr_repository_arn" {
  value = aws_ecr_repository.app.arn
}

output "runtime_secret_name" {
  value = aws_secretsmanager_secret.runtime.name
}

output "runtime_secret_arn" {
  value = aws_secretsmanager_secret.runtime.arn
}
