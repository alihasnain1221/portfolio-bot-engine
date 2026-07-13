"""Detect tech stack + maturity signals for a repo from its manifests/layout.

Reads only manifests, config files, and the README -- never source code bodies.
The git lens already excludes node_modules/venv, but we also guard here.

Testing/QA and IaC are treated as first-class signature skills (they are
prominent on the resume: E2E regressions, unit tests, test stories,
Terraform/HashiCorp) -- detected from BOTH dependencies and config files, with
dedicated signal flags (unit_tests / e2e_tests / iac).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

_SKIP_DIRS = {"node_modules", ".git", "dist", "build", ".next", "venv", ".venv",
              "__pycache__", "vendor", "coverage", ".terraform"}

# --- Dependency name -> canonical framework/tool tag -----------------------
_DEP_FRAMEWORKS = {
    # web / app frameworks
    "react": "React", "next": "Next.js", "vue": "Vue", "nuxt": "Nuxt",
    "@angular/core": "Angular", "svelte": "Svelte",
    "express": "Express", "@nestjs/core": "Nest.js", "fastify": "Fastify",
    "koa": "Koa", "react-native": "React Native", "expo": "Expo",
    "redux": "Redux", "redux-saga": "Redux Saga", "@reduxjs/toolkit": "Redux Toolkit",
    "@apollo/server": "GraphQL", "graphql": "GraphQL", "apollo-server": "GraphQL",
    # data / infra clients
    "mongoose": "MongoDB", "pg": "PostgreSQL", "mysql2": "MySQL", "mysql": "MySQL",
    "prisma": "Prisma", "typeorm": "TypeORM", "sequelize": "Sequelize",
    "socket.io": "WebSockets", "@grpc/grpc-js": "gRPC",
    "firebase": "Firebase", "firebase-admin": "Firebase", "firebase-functions": "Firebase Functions",
    "aws-sdk": "AWS", "@aws-sdk/client-s3": "AWS", "serverless": "Serverless Framework",
    "bullmq": "BullMQ", "ioredis": "Redis", "redis": "Redis",
    "stripe": "Stripe", "ethers": "ethers.js", "web3": "web3.js",
    # AI / LLM
    "openai": "OpenAI", "@anthropic-ai/sdk": "Anthropic", "langchain": "LangChain",
    "@langchain/core": "LangChain", "faiss-node": "FAISS",
    "@pinecone-database/pinecone": "Pinecone",
    # build / monorepo / styling
    "turbo": "Turborepo", "nx": "Nx", "vite": "Vite", "webpack": "Webpack",
    "tailwindcss": "Tailwind",
}

# JS/TS testing & QA libraries (framework tag, is_e2e)
_DEP_TESTING = {
    "jest": ("Jest", False),
    "@playwright/test": ("Playwright", True),
    "playwright": ("Playwright", True),
    "cypress": ("Cypress", True),
    "vitest": ("Vitest", False),
    "mocha": ("Mocha", False),
    "chai": ("Chai", False),
    "jasmine": ("Jasmine", False),
    "@testing-library/react": ("React Testing Library", False),
    "@testing-library/react-native": ("RN Testing Library", False),
    "@testing-library/jest-dom": ("Testing Library", False),
    "supertest": ("Supertest", False),
    "enzyme": ("Enzyme", False),
    "puppeteer": ("Puppeteer", True),
    "nightwatch": ("Nightwatch", True),
    "detox": ("Detox", True),
    "@cucumber/cucumber": ("Cucumber", True),
    "cucumber": ("Cucumber", True),
    "@wdio/cli": ("WebdriverIO", True),
    "webdriverio": ("WebdriverIO", True),
    "karma": ("Karma", False),
    "ava": ("AVA", False),
    "sinon": ("Sinon", False),
    "@storybook/react": ("Storybook", False),
    "storybook": ("Storybook", False),
    "@testing-library/user-event": ("Testing Library", False),
    "msw": ("Mock Service Worker", False),
}

# JS/TS IaC / HashiCorp tooling
_DEP_IAC = {
    "cdktf": "CDK for Terraform",
    "@cdktf/provider-aws": "CDK for Terraform",
    "pulumi": "Pulumi",
    "@pulumi/pulumi": "Pulumi",
    "aws-cdk-lib": "AWS CDK",
}

# Python deps (framework tag, category) -- category in {"fw","test","iac"}
_PY_DEPS = {
    "fastapi": ("FastAPI", "fw"), "flask": ("Flask", "fw"), "django": ("Django", "fw"),
    "langchain": ("LangChain", "fw"), "openai": ("OpenAI", "fw"), "anthropic": ("Anthropic", "fw"),
    "transformers": ("Transformers", "fw"), "sentence-transformers": ("Sentence-Transformers", "fw"),
    "faiss-cpu": ("FAISS", "fw"), "faiss": ("FAISS", "fw"), "torch": ("PyTorch", "fw"),
    "scikit-learn": ("scikit-learn", "fw"), "pandas": ("pandas", "fw"), "numpy": ("numpy", "fw"),
    "celery": ("Celery", "fw"), "sqlalchemy": ("SQLAlchemy", "fw"), "pydantic": ("Pydantic", "fw"),
    "scrapy": ("Scrapy", "fw"), "apify-client": ("Apify", "fw"),
    "pytest": ("pytest", "test"), "playwright": ("Playwright", "test"),
    "selenium": ("Selenium", "test"), "nose": ("nose", "test"), "tox": ("tox", "test"),
    "locust": ("Locust", "test"), "behave": ("behave", "test"), "unittest2": ("unittest", "test"),
    "pulumi": ("Pulumi", "iac"),
}

# Config-file basenames -> (framework tag, category)  category in {"test","iac","fw"}
_CONFIG_FILES = {
    "playwright.config.ts": ("Playwright", "e2e"),
    "playwright.config.js": ("Playwright", "e2e"),
    "cypress.config.ts": ("Cypress", "e2e"),
    "cypress.config.js": ("Cypress", "e2e"),
    "cypress.json": ("Cypress", "e2e"),
    "wdio.conf.js": ("WebdriverIO", "e2e"),
    "wdio.conf.ts": ("WebdriverIO", "e2e"),
    "jest.config.js": ("Jest", "unit"),
    "jest.config.ts": ("Jest", "unit"),
    "vitest.config.ts": ("Vitest", "unit"),
    "vitest.config.js": ("Vitest", "unit"),
    "karma.conf.js": ("Karma", "unit"),
    "terragrunt.hcl": ("Terragrunt", "iac"),
    "main.tf": ("Terraform", "iac"),
    "versions.tf": ("Terraform", "iac"),
    "providers.tf": ("Terraform", "iac"),
    "serverless.yml": ("Serverless Framework", "fw"),
    "serverless.yaml": ("Serverless Framework", "fw"),
}

# HashiCorp product hints inside .hcl / config content.
_HASHICORP_HINTS = ("vault", "consul", "nomad", "packer", "terraform")


@dataclass
class Stack:
    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    testing: list[str] = field(default_factory=list)      # test/QA tools specifically
    iac: list[str] = field(default_factory=list)          # IaC/HashiCorp tools
    tags: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    readme_text: str = ""            # internal grounding only, never deployed raw
    key_files: list[str] = field(default_factory=list)


def _walk(root: Path, max_depth: int = 4):
    """Yield files up to max_depth, pruning heavy dirs DURING traversal
    (so we never descend into node_modules / long broken paths)."""
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        depth = len(Path(dirpath).parts) - base
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        if depth >= max_depth:
            dirnames[:] = []          # stop descending past the depth cap
        for f in filenames:
            yield Path(dirpath) / f


def _read(p: Path, cap: int = 200_000) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:cap]
    except Exception:
        return ""


def _pkg_deps(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    deps: list[str] = []
    for k in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        deps.extend((data.get(k) or {}).keys())
    return deps


def _py_dep_present(dep: str, body: str) -> bool:
    return re.search(rf"(^|[^a-z0-9_.-]){re.escape(dep)}([^a-z0-9_.-]|$)", body) is not None


def detect(repo: Path) -> Stack:
    s = Stack()
    langs: set[str] = set()
    fw: set[str] = set()
    testing: set[str] = set()
    iac: set[str] = set()
    tags: set[str] = set()
    sig = {"unit_tests": False, "e2e_tests": False, "iac": False, "ci": False,
           "readme": False, "docker": False, "serverless": False}
    dir_names: set[str] = set()

    files = list(_walk(repo))
    names = {p.name.lower() for p in files}

    for p in files:
        name = p.name.lower()
        suffix = p.suffix.lower()
        parent = p.parent.name.lower()
        dir_names.add(parent)

        # languages by extension
        if suffix in (".ts", ".tsx"):
            langs.add("TypeScript")
        elif suffix in (".js", ".jsx", ".mjs", ".cjs"):
            langs.add("JavaScript")
        elif suffix == ".py":
            langs.add("Python")
        elif suffix == ".go":
            langs.add("Go")
        elif suffix == ".java":
            langs.add("Java")
        elif suffix == ".rb":
            langs.add("Ruby")
        elif suffix == ".vue":
            langs.add("JavaScript"); fw.add("Vue")
        elif suffix in (".tf", ".tfvars") or (suffix == ".json" and name.endswith(".tf.json")):
            langs.add("HCL"); iac.add("Terraform"); sig["iac"] = True
        elif suffix == ".hcl":
            langs.add("HCL")
            body = _read(p, 20_000).lower()
            for prod in _HASHICORP_HINTS:
                if prod in body or prod in name:
                    iac.add(prod.capitalize() if prod != "terraform" else "Terraform")
                    sig["iac"] = True

        # spec/test files by naming convention
        if re.search(r"(\.|_)(test|spec)\.[jt]sx?$", name) or name.endswith("_test.py") \
                or name.startswith("test_") and suffix == ".py":
            sig["unit_tests"] = True

        # manifests
        if name == "package.json" and "node_modules" not in str(p):
            langs.add("JavaScript")
            for dep in _pkg_deps(_read(p, 80_000)):
                d = dep.lower()
                if d in _DEP_FRAMEWORKS:
                    fw.add(_DEP_FRAMEWORKS[d])
                if d in _DEP_TESTING:
                    tag, is_e2e = _DEP_TESTING[d]
                    testing.add(tag)
                    sig["e2e_tests" if is_e2e else "unit_tests"] = True
                if d in _DEP_IAC:
                    iac.add(_DEP_IAC[d]); sig["iac"] = True
        elif name in ("requirements.txt", "pyproject.toml", "pipfile", "setup.py", "setup.cfg"):
            langs.add("Python")
            body = _read(p, 60_000).lower()
            for dep, (tag, cat) in _PY_DEPS.items():
                if _py_dep_present(dep, body):
                    if cat == "test":
                        testing.add(tag)
                        sig["e2e_tests" if tag in ("Playwright", "Selenium") else "unit_tests"] = True
                    elif cat == "iac":
                        iac.add(tag); sig["iac"] = True
                    else:
                        fw.add(tag)
        elif name == "go.mod":
            langs.add("Go")

        # config-file based detection (testing/iac/fw)
        if name in _CONFIG_FILES:
            tag, cat = _CONFIG_FILES[name]
            if cat == "e2e":
                testing.add(tag); sig["e2e_tests"] = True
            elif cat == "unit":
                testing.add(tag); sig["unit_tests"] = True
            elif cat == "iac":
                iac.add(tag); sig["iac"] = True
            else:
                fw.add(tag)

        if name in ("dockerfile", "docker-compose.yml", "docker-compose.yaml"):
            sig["docker"] = True; tags.add("Docker")
        elif name in ("serverless.yml", "serverless.yaml"):
            sig["serverless"] = True; fw.add("Serverless Framework")
        elif name.startswith("readme"):
            sig["readme"] = True
            if not s.readme_text:
                s.readme_text = _read(p, config.README_MAX_BYTES)

    # directory-based signals
    if "cypress" in dir_names or "e2e" in dir_names or "playwright" in dir_names \
            or "tests-e2e" in dir_names:
        sig["e2e_tests"] = True
    if {"__tests__", "tests", "test", "spec"} & dir_names:
        sig["unit_tests"] = True
    if ".storybook" in dir_names:
        testing.add("Storybook")
    if (repo / ".github" / "workflows").exists():
        sig["ci"] = True; tags.add("GitHub Actions")
    if names & {"bitbucket-pipelines.yml", ".gitlab-ci.yml", "buildspec.yml", "cloudbuild.yaml"}:
        sig["ci"] = True; tags.add("CI/CD")

    if sig["e2e_tests"]:
        tags.add("E2E Testing")
    if sig["unit_tests"]:
        tags.add("Unit Testing")
    if sig["iac"]:
        tags.add("IaC")

    s.languages = sorted(langs)
    s.frameworks = sorted(fw)
    s.testing = sorted(testing)
    s.iac = sorted(iac)
    s.tags = sorted(tags)
    s.signals = sig
    s.key_files = [
        str(p.relative_to(repo)).replace("\\", "/")
        for p in files
        if p.name.lower() in ("readme.md", "package.json", "requirements.txt",
                              "serverless.yml", "docker-compose.yml", "main.tf",
                              "playwright.config.ts", "cypress.config.ts")
    ][:12]
    return s
