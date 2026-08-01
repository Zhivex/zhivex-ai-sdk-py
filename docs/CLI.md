# Zhivex CLI And Local Playground

Version `0.16.0` adds the beta `zhivex` command for trusted local agent modules.

An agent reference uses `module:attribute`. Importing it executes that module's Python code, like loading an ASGI application, so only load reviewed local code.

```bash
zhivex inspect my_app.agents:support_agent
zhivex run my_app.agents:support_agent --prompt "Draft a reply"
zhivex run my_app.agents:support_agent --prompt "Draft a reply" --json
zhivex eval my_app.agents:support_agent --dataset evals/support.json
```

Install API dependencies to serve an agent:

```bash
pip install "zhivex-ai-sdk[api]"
zhivex serve my_app.agents:support_agent --model-alias support
```

The Responses-compatible server binds to `127.0.0.1:8000` by default. A2A uses the official SDK extra:

```bash
pip install "zhivex-ai-sdk[a2a]"
zhivex serve my_app.agents:support_agent \
  --protocol a2a \
  --public-url https://agents.example.com \
  --agent-version 1.0.0
```

For a local browser UI:

```bash
zhivex playground my_app.agents:support_agent --model-alias default
```

Open `http://127.0.0.1:8000`. The playground uses the same `/v1/responses` streaming endpoint as the host.

The CLI server and playground are development conveniences. They do not add authentication, TLS, distributed rate limiting, tenant-scoped persistence, secret management, audit retention, or an approval UI. Keep loopback binding for local work. Production deployments should create the app in application code, provide authorization, and run it through the organization's normal ASGI, gateway, IAM, network, and observability controls.
