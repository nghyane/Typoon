export interface HeaderBack {
	label: string;
	href?: string;
}

class HeaderStore {
	back = $state<HeaderBack | null>(null);

	set(back: HeaderBack | null): void {
		this.back = back;
	}
}

export const headerStore = new HeaderStore();
