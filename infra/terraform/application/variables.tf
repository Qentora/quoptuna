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

variable "domain_name" {
  type = string
}

variable "route53_zone_id" {
  type = string
}

variable "instance_type" {
  type    = string
  default = "t3.large"
}

variable "root_volume_size" {
  type    = number
  default = 50
}

variable "artifact_bucket" {
  type = string
}

variable "artifact_bucket_arn" {
  type = string
}

variable "ecr_repository_name" {
  type = string
}

variable "ecr_repository_url" {
  type = string
}

variable "ecr_repository_arn" {
  type = string
}

variable "runtime_secret_name" {
  type = string
}

variable "runtime_secret_arn" {
  type = string
}
