type InstallOutcome = 'accepted' | 'dismissed';

interface BeforeInstallPromptEvent extends Event {
	readonly platforms: string[];
	readonly userChoice: Promise<{ outcome: InstallOutcome; platform: string }>;
	prompt(): Promise<void>;
}

interface InstallState {
	readonly supported: boolean;
	readonly ready: boolean;
	readonly installed: boolean;
	readonly lastOutcome: InstallOutcome | null;
}

const installedFlagKey = 'typoon.pwa.installed';

class PwaInstallPrompt {
	state = $state<InstallState>({ supported: false, ready: false, installed: false, lastOutcome: null });
	#prompt: BeforeInstallPromptEvent | null = null;
	#initialized = false;

	init(): void {
		if (this.#initialized || typeof window === 'undefined') return;
		this.#initialized = true;
		this.state = { ...this.state, supported: 'serviceWorker' in navigator, installed: isInstalled() };

		window.addEventListener('beforeinstallprompt', (event) => {
			event.preventDefault();
			this.#prompt = event as BeforeInstallPromptEvent;
			// A fresh install prompt means the app is not currently installed; clear
			// any stale persisted flag so the button offers installation again.
			rememberInstalled(false);
			this.state = { ...this.state, ready: true, installed: isInstalled(), lastOutcome: null };
		});

		window.addEventListener('appinstalled', () => {
			this.#prompt = null;
			rememberInstalled(true);
			this.state = { ...this.state, ready: false, installed: true, lastOutcome: 'accepted' };
		});

		void this.#detectInstalled();
	}

	async install(): Promise<void> {
		if (!this.#prompt) return;
		const prompt = this.#prompt;
		this.#prompt = null;
		this.state = { ...this.state, ready: false };
		await prompt.prompt();
		const choice = await prompt.userChoice.catch(() => ({ outcome: 'dismissed' as const, platform: '' }));
		const installed = isInstalled() || choice.outcome === 'accepted';
		if (installed) rememberInstalled(true);
		this.state = { ...this.state, installed, lastOutcome: choice.outcome };
	}

	// Detect an install that happened in another tab/session: standalone display,
	// a persisted flag, or the related-apps API (Chromium, current-origin app).
	async #detectInstalled(): Promise<void> {
		if (isInstalled() || readInstalledFlag()) {
			this.state = { ...this.state, installed: true };
			return;
		}
		if (await hasInstalledRelatedApp()) {
			rememberInstalled(true);
			this.state = { ...this.state, installed: true };
		}
	}
}

export const pwaInstall = new PwaInstallPrompt();

function isInstalled(): boolean {
	const navigatorWithStandalone = navigator as Navigator & { standalone?: boolean };
	return window.matchMedia('(display-mode: standalone)').matches || navigatorWithStandalone.standalone === true;
}

function readInstalledFlag(): boolean {
	try {
		return localStorage.getItem(installedFlagKey) === '1';
	} catch {
		return false;
	}
}

function rememberInstalled(installed: boolean): void {
	try {
		if (installed) localStorage.setItem(installedFlagKey, '1');
		else localStorage.removeItem(installedFlagKey);
	} catch {
		// ignore storage failures (private mode, quota)
	}
}

interface RelatedApplication {
	readonly platform?: string;
}

async function hasInstalledRelatedApp(): Promise<boolean> {
	const nav = navigator as Navigator & {
		getInstalledRelatedApps?: () => Promise<RelatedApplication[]>;
	};
	if (typeof nav.getInstalledRelatedApps !== 'function') return false;
	try {
		const apps = await nav.getInstalledRelatedApps();
		return apps.some((app) => app.platform === 'webapp');
	} catch {
		return false;
	}
}
