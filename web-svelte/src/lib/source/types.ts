export type ParseMode = 'html' | 'json';
export type Selector = string;

export interface HttpRequest {
	method?: 'GET' | 'POST';
	url: string;
	headers?: Record<string, string>;
	body?: string;
	parse: ParseMode;
}

export interface Pagination {
	type: 'page' | 'offset' | 'cursor';
	pageSize: number;
	/** Optional selector on the response root; loop stops when it resolves falsy
	 *  (empty, "0", "false", "null"). Lets sources that clamp overflow pages
	 *  signal the last page instead of relying on a wasted clamp request. */
	hasMore?: Selector;
}

export interface Filter {
	id: string;
	label: string;
	type: 'select' | 'multi';
	/** How selected option `param`s reach the request. Default `'param'`.
	 *  - `param`: query fragments joined with `&`, substituted at `{filterParams}`.
	 *  - `path`:  a single value substituted at `{filterPath}` (single-select; always
	 *             keeps one option active so the path never goes empty).
	 *  - `query`: terms folded into the `q` value (space-joined), e.g. nhentai tags. */
	inject?: 'param' | 'path' | 'query';
	options: FilterOption[];
}

export interface FilterOption {
	id: string;
	label: string;
	param: string;
	value?: string;
	nsfw?: boolean;
}

export interface BrowseEndpoint extends HttpRequest {
	pagination?: Pagination;
	list: Selector;
	fields: {
		url: Selector;
		title: Selector;
		cover?: Selector;
	};
	rootExtras?: Record<string, Selector>;
	extras?: Record<string, Selector>;
	keepIf?: Record<string, Selector>;
}

export interface MangaEndpoint extends HttpRequest {
	extract?: string;
	fields: {
		title: Selector;
		cover?: Selector;
		description?: Selector;
		author?: Selector;
		status?: Selector;
		updatedAt?: Selector;
		availableLangs?: Selector;
	};
	rootExtras?: Record<string, Selector>;
	extras?: Record<string, Selector>;
	chapters?: ChapterListSpec;
}

export interface ChaptersApiEndpoint extends HttpRequest {
	pagination?: Pagination;
	list: Selector;
	fields: ChapterFields;
	extras?: Record<string, Selector>;
	chapterNumberNorm?: ChapterNumberNorm;
	keepIf?: Record<string, Selector>;
}

export interface ChapterListSpec {
	list: Selector;
	fields: ChapterFields;
	chapterNumberNorm?: ChapterNumberNorm;
}

export interface ChapterFields {
	url: Selector;
	number?: Selector;
	title?: Selector;
	date?: Selector;
	language?: Selector;
	label?: Selector;
	scanlator?: Selector;
}

export interface ChapterNumberNorm {
	input?: 'number' | 'label';
	patterns?: string[];
	default?: 'slug' | 'empty' | 'verbatim';
	postprocess?: ('lowercase' | 'trim' | 'stripLeadingZeros')[];
}

export interface ChapterEndpoint extends HttpRequest {
	extract?: string;
	list?: Selector;
	fields?: { url: Selector };
	pageHeaders?: Record<string, string>;
	rootExtras?: Record<string, Selector>;
	pages?: {
		extras: Record<string, Selector>;
		iterate?: string;
		count?: string;
		template: string;
	};
}

export interface SourceManifest {
	id: string;
	name: string;
	host: string;
	homepage?: string;
	languages: string[];
	kind?: 'external' | 'internal';
	nsfw?: boolean;
	version: string;
	adapter?: string;
	requestHeaders?: Record<string, string>;
	imageHeaders?: Record<string, string>;
	authRequired?: 'none' | 'cookie' | 'token';
	cookieNames?: string[];
	icon?: string;
	accent?: string;
	defaults?: Record<string, string | string[]>;
	filters?: Filter[];
	chapterNumberNorm?: ChapterNumberNorm;
	endpoints?: {
		shelves: Shelf[];
		search?: BrowseEndpoint;
		manga: MangaEndpoint;
		chaptersApi?: ChaptersApiEndpoint;
		chapter: ChapterEndpoint;
	};
}

export interface Shelf {
	id: string;
	label: string;
	hint?: string;
	/** Filters scoped to this shelf only. When present they override the
	 *  source-level `manifest.filters` while this shelf is active. */
	filters?: Filter[];
	endpoint: BrowseEndpoint;
}

export interface MangaSummary {
	id: string;
	url: string;
	title: string;
	cover: string | null;
	coverHeaders?: Record<string, string>;
}

export interface MangaChapterRef {
	id: string;
	url: string;
	label: string;
	number: string;
	numberNorm: string;
	title: string | null;
	date: string | null;
	language: string | null;
	scanlator: string | null;
}

export interface MangaDetail extends MangaSummary {
	description: string | null;
	author: string | null;
	status: string | null;
	availableLanguages: string[] | null;
	chapters: MangaChapterRef[];
}

export interface ChapterPages {
	url: string;
	pages: string[];
	tokens?: string[];
	pageHeaders?: Record<string, string>;
}

export interface BrowseArgs {
	q?: string;
	page?: number;
	/** Query fragments (`&a&b`) for `inject: 'param'` filters, spliced at `{filterParams}`. */
	filterParams?: string;
	/** Single path segment for an `inject: 'path'` filter, spliced at `{filterPath}`. */
	filterPath?: string;
	/** Extra `q` terms for `inject: 'query'` filters; folded into the search query. */
	filterQuery?: string;
	filterState?: Record<string, string | string[]>;
	userCookies?: Record<string, string>;
}

export type SourceOrigin = 'bundled' | 'repo' | 'file';

export interface InstalledSource {
	manifest: SourceManifest;
	origin: SourceOrigin;
	repoUrl?: string;
	author?: string;
	installedAt: number;
	enabled: boolean;
}
