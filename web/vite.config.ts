import { defineConfig } from 'vite'

// No backend and no external fetches — the simulator and filter are the WASM module, so the
// whole app is static. Base path is set on the command line for the showcase embed.
export default defineConfig({
  build: {
    target: 'es2022',
  },
})
