export interface Brand {
	name: string;
	monogram: string;
	logoUrl: string;
}

export const BRAND: Brand = {
	name: import.meta.env.VITE_BRAND_NAME ?? 'Hội Mê Truyện',
	monogram: import.meta.env.VITE_BRAND_MONOGRAM ?? 'HMT',
	logoUrl: import.meta.env.VITE_BRAND_LOGO_URL ?? '/brand/logo.webp',
};
