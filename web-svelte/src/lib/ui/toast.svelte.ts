import type { Info } from 'lucide-svelte';

// A lucide-svelte icon component (matches the `icon: typeof Clock` convention
// used elsewhere, e.g. WorkShelf).
type IconComponent = typeof Info;

// App-wide toast (snackbar) store. Toasts stack at the bottom-right, auto-dismiss
// after `duration` ms (0 = sticky), and can carry one action button. Keep the
// surface intentionally small: a title, an optional description, one action, and
// a dismiss — anything richer belongs in a modal, not a transient toast.

export type ToastVariant = 'default' | 'success' | 'error' | 'info';

export interface ToastAction {
	readonly label: string;
	/** Navigate here when pressed (client-side goto). */
	readonly href?: string;
	/** Or run this. Mutually exclusive with href; both dismiss the toast first. */
	readonly onClick?: () => void;
}

export interface ToastOptions {
	readonly title: string;
	readonly description?: string;
	readonly variant?: ToastVariant;
	readonly action?: ToastAction;
	/** A cover/thumbnail to lead with instead of an icon — far more recognizable
	 *  than a generic glyph for content-specific toasts (e.g. a manga cover). */
	readonly cover?: { src: string | null; headers?: Record<string, string> | null; title?: string | null };
	/** Override the variant's default icon (used only when no `cover` is given). */
	readonly icon?: IconComponent;
	/** ms before auto-dismiss; 0 keeps it until the user dismisses. Default 5000. */
	readonly duration?: number;
	/** When set, replaces any live toast sharing the key instead of stacking a
	 *  duplicate (e.g. only ever one "continue reading" toast at a time). */
	readonly key?: string;
}

export interface Toast extends ToastOptions {
	readonly id: number;
}

// Cap the stack so a burst can't bury the screen; oldest falls off first.
const MAX_VISIBLE = 3;
const DEFAULT_DURATION = 5000;

class ToastStore {
	items = $state<Toast[]>([]);
	#seq = 0;
	#timers = new Map<number, ReturnType<typeof setTimeout>>();

	show(opts: ToastOptions): number {
		const id = ++this.#seq;
		const toast: Toast = { variant: 'default', duration: DEFAULT_DURATION, ...opts, id };

		// Replace an existing keyed toast rather than stacking another copy.
		let next = opts.key ? this.items.filter((t) => t.key !== opts.key) : [...this.items];
		if (opts.key) this.#clearKeyTimers(opts.key);
		next.push(toast);
		while (next.length > MAX_VISIBLE) {
			const dropped = next.shift();
			if (dropped) this.#clearTimer(dropped.id);
		}
		this.items = next;
		this.#arm(toast);
		return id;
	}

	dismiss(id: number): void {
		this.#clearTimer(id);
		this.items = this.items.filter((t) => t.id !== id);
	}

	/** Pause auto-dismiss (hover/focus) and re-arm it (pointer/focus leaves). */
	pause(id: number): void {
		this.#clearTimer(id);
	}
	resume(toast: Toast): void {
		this.#arm(toast);
	}

	#arm(toast: Toast): void {
		if (!toast.duration || typeof setTimeout === 'undefined') return;
		this.#clearTimer(toast.id);
		this.#timers.set(toast.id, setTimeout(() => this.dismiss(toast.id), toast.duration));
	}
	#clearTimer(id: number): void {
		const handle = this.#timers.get(id);
		if (handle !== undefined) {
			clearTimeout(handle);
			this.#timers.delete(id);
		}
	}
	#clearKeyTimers(key: string): void {
		for (const t of this.items) if (t.key === key) this.#clearTimer(t.id);
	}
}

export const toast = new ToastStore();
