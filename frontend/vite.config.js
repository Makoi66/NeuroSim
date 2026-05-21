import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import pkg from './package.json' with { type: 'json' }

const APP_VERSION = `v${pkg.version.split('.').slice(0, 2).join('.')}`

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'inject-app-version',
      transformIndexHtml: (html) => html.replace(/%APP_VERSION%/g, APP_VERSION),
    },
  ],
  define: {
    __APP_VERSION__: JSON.stringify(APP_VERSION),
  },
})
