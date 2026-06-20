import { ehentaiAdapter } from './ehentai';
import { hentaifoxAdapter } from './hentaifox';
import { hitomiAdapter } from './hitomi';
import { nhentaiAdapter } from './nhentai';
import type { SourceAdapter } from './types';

const REGISTRY: Record<string, SourceAdapter> = {
	ehentai: ehentaiAdapter,
	hentaifox: hentaifoxAdapter,
	hitomi: hitomiAdapter,
	nhentai: nhentaiAdapter,
};

export function getAdapter(id: string): SourceAdapter | null {
	return REGISTRY[id] ?? null;
}

export type { SourceAdapter };
