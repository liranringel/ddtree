import { defineConfig } from "astro/config";

const repository = process.env.GITHUB_REPOSITORY ?? "liranringel/ddtree";
const [, repoName] = repository.split("/");
const isGitHubActions = process.env.GITHUB_ACTIONS === "true";

export default defineConfig({
  site: "https://liranringel.github.io",
  base: isGitHubActions && repoName ? `/${repoName}` : undefined,
  server: {
    host: true,
    allowedHosts: [".ngrok-free.app"],
  },
});
