import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import { resetLiveResultsCacheForTests } from "../useLiveResults";

beforeEach(() => {
  resetLiveResultsCacheForTests();
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("Live results unavailable in tests")));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
