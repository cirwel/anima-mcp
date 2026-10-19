# Security Policy

## Reporting a vulnerability

Please use [GitHub Private Vulnerability Reporting](https://github.com/cirwel/anima-mcp/security/advisories/new) to report security issues. The report goes privately to the maintainer, and a fix can be coordinated before public disclosure.

Do not open a public issue or pull request for a security bug.

If private reporting is unavailable to you (for example, you have no GitHub account), open a public issue that says only "contact requested: security", with no details, and I will reach out to set up a private channel.

## Scope

anima-mcp is a single-operator MCP server for one embodied Raspberry Pi agent. In-scope issues include:

- Authentication or authorization bypasses on the MCP transport, the REST API, or the OAuth flow
- Bypassing the admin gate on destructive tools (`git_pull`, `deploy_from_github`, `system_service`, `system_power`, `fix_ssh_port`, `setup_tailscale`)
- Injection or remote code execution through tool inputs
- Credentials, tokens, or other secrets exposed in the repository or in server responses

Out of scope:

- Issues that require existing root or local-shell access to the host
- Vulnerabilities in dependencies; please report those upstream first

## Supported versions

Only the current `main` branch receives security fixes.

## Response

I aim to acknowledge reports within 3 business days and to coordinate disclosure within 90 days of a confirmed report, sooner for issues under active exploitation.
