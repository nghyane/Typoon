export function registerServiceWorker(): void {
	if (typeof window === 'undefined' || !('serviceWorker' in navigator)) return;
	const run = () => {
		void navigator.serviceWorker.register('/service-worker.js', { type: 'module' }).then((registration) => registration.update()).catch(() => undefined);
	};
	const idle = window as Window & { requestIdleCallback?: (callback: () => void, options?: { timeout?: number }) => number };
	if (typeof idle.requestIdleCallback === 'function') idle.requestIdleCallback(run, { timeout: 3_000 });
	else window.setTimeout(run, 1_000);
}
