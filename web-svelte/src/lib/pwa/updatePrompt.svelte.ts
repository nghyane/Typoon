import { browser, version } from '$app/environment';
import { updated } from '$app/state';
import { track } from '$lib/analytics/client';

const APPLY_STALL_MS = 8_000;

type UpdatePhase = 'idle' | 'applying' | 'stalled';

interface UpdateState {
	readonly phase: UpdatePhase;
	readonly error: string | null;
}

class PwaUpdatePrompt {
	state = $state<UpdateState>({ phase: 'idle', error: null });

	#initialized = false;
	#dismissed = false;
	#tracked = false;

	/** Banner visibility: SvelteKit detected a new version and user hasn't dismissed it. */
	get available(): boolean {
		return updated.current && !this.#dismissed;
	}

	init(): void {
		if (!browser || this.#initialized) return;
		this.#initialized = true;
		// SvelteKit already polls on its own pollInterval. Add focus/online
		// nudges so a returning user sees the update sooner.
		document.addEventListener('visibilitychange', () => {
			if (document.visibilityState === 'visible') void this.check();
		});
		window.addEventListener('online', () => { void this.check(); });
		$effect.root(() => {
			$effect(() => {
				if (updated.current && !this.#tracked) {
					this.#tracked = true;
					track('app_update_available', { current_version: version });
				}
			});
		});
	}

	async check(): Promise<void> {
		if (!browser || this.state.phase === 'applying') return;
		await updated.check();
	}

	dismiss(): void {
		this.#dismissed = true;
		this.state = { phase: 'idle', error: null };
		track('app_update_dismiss', { current_version: version });
	}

	apply(): void {
		if (!browser || this.state.phase === 'applying') return;
		this.state = { phase: 'applying', error: null };
		track('app_update_apply', { current_version: version });

		window.setTimeout(() => {
			if (this.state.phase === 'applying') {
				this.state = {
					phase: 'stalled',
					error: 'Kết nối hoặc cache trình duyệt đang phản hồi chậm.',
				};
			}
		}, APPLY_STALL_MS);

		void activateWaitingWorker()
			.then(() => window.setTimeout(() => window.location.reload(), 120))
			.catch((error: unknown) => {
				this.state = {
					phase: 'stalled',
					error: error instanceof Error ? error.message : String(error),
				};
			});
	}

	forceReload(): void {
		track('app_update_force_reload', { current_version: version });
		window.location.reload();
	}
}

export const pwaUpdate = new PwaUpdatePrompt();

async function activateWaitingWorker(): Promise<void> {
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
