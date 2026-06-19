import { ApiError, authApi, type SessionUser } from './api';

export type SessionState =
	| { status: 'loading'; user: null }
	| { status: 'unauthenticated'; user: null }
	| { status: 'authenticated'; user: SessionUser }
	| { status: 'error'; user: null; error: Error };

class AuthSession {
	state = $state<SessionState>({ status: 'loading', user: null });
	#pending: Promise<void> | null = null;

	load(): Promise<void> {
		if (this.#pending) return this.#pending;
		this.state = { status: 'loading', user: null };
		this.#pending = authApi.getSession()
			.then((user) => { this.state = { status: 'authenticated', user }; })
			.catch((err) => {
				if (err instanceof ApiError && err.status === 401) {
					this.state = { status: 'unauthenticated', user: null };
					return;
				}
				this.state = { status: 'error', user: null, error: err instanceof Error ? err : new Error(String(err)) };
			})
			.finally(() => { this.#pending = null; });
		return this.#pending;
	}

	async refresh(): Promise<void> {
		this.#pending = null;
		await this.load();
	}

	async signOut(): Promise<void> {
		try { await authApi.logout(); } catch {}
		this.state = { status: 'unauthenticated', user: null };
	}

	setAuthenticated(user: SessionUser): void {
		this.state = { status: 'authenticated', user };
	}
}

export const session = new AuthSession();
