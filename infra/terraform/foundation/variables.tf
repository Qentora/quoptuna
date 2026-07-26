variable "aws_region" {
  type = string
}

variable "environment" {
  type = string

  validation {
    condition     = contains(["dev", "production"], var.environment)
    error_message = "environment must be dev or production"
  }
}

variable "project_name" {
  type    = string
  default = "quoptuna"
}

variable "archive_artifacts_after_days" {
  type    = number
  default = 90
}
