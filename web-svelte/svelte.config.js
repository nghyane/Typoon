import adapter from '@sveltejs/adapter-cloudflare';

// Single source of truth for the app version. Cached in process.env so every
// build pass (client + server) within one build reuses the same value.
// SvelteKit serves this at /_app/version.json and polls it to detect updates.
const version = (process.env.TYPOON_APP_VERSION ??= Date.now().toString());

/** @type {import('@sveltejs/kit').Config} */
export default {
	compilerOptions: {
		runes: ({ filename }) =>
			filename.split(/[/\\]/).includes('node_modules') ? undefined : true,
	},
	kit: {
		adapter: adapter(),
		version: {
			name: version,
			pollInterval: 5 * 60_000,
		},
	},
};
