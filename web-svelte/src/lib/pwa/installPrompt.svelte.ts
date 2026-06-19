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
			this.state = { ...this.state, ready: true, installed: isInstalled(), lastOutcome: null };
		});

		window.addEventListener('appinstalled', () => {
			this.#prompt = null;
			this.state = { ...this.state, ready: false, installed: true, lastOutcome: 'accepted' };
		});
	}

	async install(): Promise<void> {
		if (!this.#prompt) return;
		const prompt = this.#prompt;
		this.#prompt = null;
		this.state = { ...this.state, ready: false };
		await prompt.prompt();
		const choice = await prompt.userChoice.catch(() => ({ outcome: 'dismissed' as const, platform: '' }));
		this.state = { ...this.state, installed: isInstalled() || choice.outcome === 'accepted', lastOutcome: choice.outcome };
	}
}

export const pwaInstall = new PwaInstallPrompt();

function isInstalled(): boolean {
	const navigatorWithStandalone = navigator as Navigator & { standalone?: boolean };
	return window.matchMedia('(display-mode: standalone)').matches || navigatorWithStandalone.standalone === true;
}
