import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import { fileURLToPath, URL } from 'node:url';

const cdnTarget = 'https://927251094806098001.discordsays.com';
const deeplTarget = 'https://927251094806098001.discordsays.com';
const repoRoot = fileURLToPath(new URL('..', import.meta.url));

// Dynamic version to force PWA update check on every deploy
const appVersion = Date.now().toString();

export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit(),
	],
	define: {
		'import.meta.env.VITE_APP_VERSION': JSON.stringify(appVersion),
	},
	server: {
		fs: {
			allow: [repoRoot],
		},
		proxy: {
			'/cdn': { target: cdnTarget, changeOrigin: true },
			'/deepl': { target: deeplTarget, changeOrigin: true, ws: true },
			'/api': { target: 'http://localhost:3000', changeOrigin: true },
		},
	},
	optimizeDeps: {
		include: ['comlink', 'chrome-lens-ocr', 'onnxruntime-web'],
	},
});
