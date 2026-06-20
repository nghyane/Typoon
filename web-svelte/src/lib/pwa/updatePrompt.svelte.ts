import { browser, version as appVersion } from '$app/environment';
import { track } from '$lib/analytics/client';

const DISMISSED_VERSION_KEY = 'typoon.dismissedUpdateVersion.v1';
const CHECK_INTERVAL_MS = 5 * 60_000;
const APPLY_STALL_MS = 8_000;

type UpdatePhase = 'idle' | 'applying' | 'stalled';

interface UpdateState {
	readonly available: boolean;
	readonly currentVersion: string;
	readonly latestVersion: string | null;
	readonly checking: boolean;
	readonly phase: UpdatePhase;
	readonly error: string | null;
}

class PwaUpdatePrompt {
	state = $state<UpdateState>({
		available: false,
		currentVersion: appVersion,
		latestVersion: null,
		checking: false,
		phase: 'idle',
		error: null,
	});

	#initialized = false;
	#trackedVersion: string | null = null;

	init(): void {
		if (!browser || this.#initialized) return;
		this.#initialized = true;
		void this.check();
		window.setInterval(() => { void this.check(); }, CHECK_INTERVAL_MS);
		document.addEventListener('visibilitychange', () => {
			if (document.visibilityState === 'visible') void this.check();
		});
		window.addEventListener('online', () => { void this.check(); });
	}

	async check(): Promise<void> {
		if (!browser || this.state.checking || this.state.phase === 'applying') return;
		this.state = { ...this.state, checking: true };
		try {
			const latestVersion = await fetchLatestVersion();
			const dismissedVersion = localStorage.getItem(DISMISSED_VERSION_KEY);
			const available = latestVersion !== appVersion && dismissedVersion !== latestVersion;
			this.state = {
				...this.state,
				latestVersion,
				available,
				checking: false,
				phase: available ? this.state.phase : 'idle',
				error: available ? this.state.error : null,
			};
			if (available && this.#trackedVersion !== latestVersion) {
				this.#trackedVersion = latestVersion;
				track('app_update_available', { current_version: appVersion, latest_version: latestVersion });
			}
		} catch {
			this.state = { ...this.state, checking: false };
		}
	}

	dismiss(): void {
		const latestVersion = this.state.latestVersion;
		if (latestVersion) localStorage.setItem(DISMISSED_VERSION_KEY, latestVersion);
		this.state = { ...this.state, available: false, phase: 'idle', error: null };
		track('app_update_dismiss', { current_version: appVersion, latest_version: latestVersion ?? '' });
	}

	apply(): void {
		if (!browser || this.state.phase === 'applying') return;
		const latestVersion = this.state.latestVersion ?? '';
		this.state = { ...this.state, phase: 'applying', error: null };
		track('app_update_apply', { current_version: appVersion, latest_version: latestVersion });

		window.setTimeout(() => {
			if (this.state.phase === 'applying') {
				this.state = {
					...this.state,
					phase: 'stalled',
					error: 'Kết nối hoặc cache trình duyệt đang phản hồi chậm.',
				};
			}
		}, APPLY_STALL_MS);

		void prepareServiceWorkerUpdate()
			.then(() => window.setTimeout(() => window.location.reload(), 120))
			.catch((error: unknown) => {
				this.state = {
					...this.state,
					phase: 'stalled',
					error: error instanceof Error ? error.message : String(error),
				};
			});
	}

	forceReload(): void {
		track('app_update_force_reload', { current_version: appVersion, latest_version: this.state.latestVersion ?? '' });
		window.location.reload();
	}
}

export const pwaUpdate = new PwaUpdatePrompt();

async function fetchLatestVersion(): Promise<string> {
	const res = await fetch(`/_app/version.json?ts=${Date.now()}`, {
		cache: 'no-store',
		headers: { accept: 'application/json' },
	});
	if (!res.ok) throw new Error(`version check failed: ${res.status}`);
	const body = await res.json() as { version?: unknown };
	if (typeof body.version !== 'string' || !body.version) throw new Error('invalid version payload');
	return body.version;
}

async function prepareServiceWorkerUpdate(): Promise<void> {
	if (!('serviceWorker' in navigator)) return;
	const registration = await withTimeout(navigator.serviceWorker.getRegistration(), 1_500);
	const updated = await withTimeout(registration?.update() ?? Promise.resolve(null), 2_000);
	const waiting = updated?.waiting ?? registration?.waiting;
	if (!waiting) return;
	const controllerChanged = waitForControllerChange(2_000);
	waiting.postMessage({ type: 'SKIP_WAITING' });
	await controllerChanged;
}

function waitForControllerChange(timeoutMs: number): Promise<void> {
	return new Promise((resolve) => {
		let done = false;
		const finish = () => {
			if (done) return;
			done = true;
			window.clearTimeout(timer);
			navigator.serviceWorker.removeEventListener('controllerchange', finish);
			resolve();
		};
		const timer = window.setTimeout(finish, timeoutMs);
		navigator.serviceWorker.addEventListener('controllerchange', finish);
	});
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T | null> {
	return new Promise((resolve) => {
		let done = false;
		const finish = (value: T | null) => {
			if (done) return;
			done = true;
			window.clearTimeout(timer);
			resolve(value);
		};
		const timer = window.setTimeout(() => finish(null), timeoutMs);
		promise.then(finish, () => finish(null));
	});
}
