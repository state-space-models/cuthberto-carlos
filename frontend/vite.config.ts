/// <reference types="vitest/config" />

import { copyFileSync, mkdirSync } from "node:fs";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { getPagesBase } from "./pages";

const tournamentData = new URL("./src/data/tournament.json", import.meta.url);
const deployedDataDirectory = new URL("./dist/data/", import.meta.url);

export default defineConfig({
  base: getPagesBase(),
  publicDir: new URL("../assets", import.meta.url).pathname,
  plugins: [
    react(),
    {
      name: "expose-tournament-data",
      closeBundle() {
        mkdirSync(deployedDataDirectory, { recursive: true });
        copyFileSync(tournamentData, new URL("tournament.json", deployedDataDirectory));
      },
    },
  ],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: true,
  },
});
