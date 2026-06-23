import { browser } from '$app/environment';

const GA_ID = (import.meta.env.VITE_GA_MEASUREMENT_ID as string | undefined)?.trim() || 'G-GG3N4DZD64';
const GA_COLLECT_URL = 'https://927251094806098001.discordsays.com/cdn/c/www.google-analytics.com/g/collect';
const CLIENT_ID_KEY = 'hmt.ga.client_id';
const SESSION_KEY = 'hmt.ga.session';
const SESSION_TIMEOUT_MS = 30 * 60 * 1000;

type AnalyticsParams = Record<string, string | number | boolean | null | undefined>;

interface AnalyticsSession {
	readonly id: number;
	readonly count: number;
	readonly lastSeen: number;
}

let initialized = false;
let eventIndex = 0;
let clientId: string | null = null;
let session: AnalyticsSession | null = null;
let isFirstVisit = false;
let isNewSession = false;

export function initAnalytics(): void {
	if (!browser || initialized) return;
	initialized = true;
	clientId = getClientId();
	session = getSession();
}

export function trackPageView(path: string, title = document.title): void {
	track('page_view', {
		page_path: path,
		page_location: window.location.href,
		page_title: title,
	});
}

export function track(eventName: string, params: AnalyticsParams = {}): void {
	if (!browser) return;
	initAnalytics();
	void sendCollectHit(eventName, compactParams(params));
}

export function trackTranslateClick(params: AnalyticsParams): void {
	track('translate_click', params);
}

export function trackSourceSelect(params: AnalyticsParams): void {
	track('source_select', params);
}

export function trackSourceOpen(params: AnalyticsParams): void {
	track('source_open', params);
}

export function trackDiscordJoinRequired(params: AnalyticsParams = {}): void {
	track('discord_join_required_shown', params);
}

async function sendCollectHit(eventName: string, params: AnalyticsParams): Promise<void> {
	const url = collectUrl(eventName, params);
	try {
		await fetch(url, { credentials: 'omit', keepalive: true });
	} catch {
		// Analytics must never affect app behavior.
	}
}

function collectUrl(eventName: string, params: AnalyticsParams): string {
	const currentSession = getSession();
	const url = new URL(GA_COLLECT_URL);
	url.searchParams.set('v', '2');
	url.searchParams.set('tid', GA_ID);
	url.searchParams.set('cid', clientId ?? getClientId());
	url.searchParams.set('en', normalizeParamName(eventName) || eventName);
	url.searchParams.set('_p', pageNonce());
	url.searchParams.set('_s', String(++eventIndex));
	url.searchParams.set('_et', '1000');
	url.searchParams.set('sid', String(currentSession.id));
	url.searchParams.set('sct', String(currentSession.count));
	url.searchParams.set('seg', '1');
	url.searchParams.set('ul', navigator.language || 'vi');
	url.searchParams.set('sr', screen.width && screen.height ? `${screen.width}x${screen.height}` : '');
	url.searchParams.set('dl', stringParam(params.page_location) ?? window.location.href);
	url.searchParams.set('dt', stringParam(params.page_title) ?? document.title);
	if (document.referrer) url.searchParams.set('dr', document.referrer);
	if (isFirstVisit) url.searchParams.set('_fv', '1');
	if (isNewSession) url.searchParams.set('_ss', '1');
	isFirstVisit = false;
	isNewSession = false;

	const pagePath = stringParam(params.page_path);
	if (pagePath) url.searchParams.set('dp', pagePath);

	for (const [key, value] of Object.entries(params)) {
		if (key === 'page_location' || key === 'page_title' || key === 'page_path') continue;
		appendEventParam(url, key, value);
	}

	return url.href;
}

function appendEventParam(url: URL, key: string, value: AnalyticsParams[string]): void {
	const name = normalizeParamName(key);
	if (!name) return;
	if (typeof value === 'number' && Number.isFinite(value)) {
		url.searchParams.set(`epn.${name}`, String(value));
		return;
	}
	url.searchParams.set(`ep.${name}`, String(value));
}

function normalizeParamName(name: string): string {
	return name.trim().replace(/[^a-zA-Z0-9_]/g, '_').slice(0, 40);
}

function stringParam(value: AnalyticsParams[string]): string | null {
	if (typeof value !== 'string') return null;
	return value.trim() || null;
}

function compactParams(params: AnalyticsParams): AnalyticsParams {
	return Object.fromEntries(Object.entries(params).filter(([, value]) => value !== undefined && value !== null));
}

function getClientId(): string {
	if (clientId) return clientId;
	const stored = readStorage(CLIENT_ID_KEY);
	if (stored) {
		clientId = stored;
		return stored;
	}
	clientId = `${randomInt()}.${Math.floor(Date.now() / 1000)}`;
	isFirstVisit = true;
	writeStorage(CLIENT_ID_KEY, clientId);
	return clientId;
}

function getSession(): AnalyticsSession {
	const now = Date.now();
	if (session && now - session.lastSeen < SESSION_TIMEOUT_MS) {
		session = { ...session, lastSeen: now };
		writeSession(session);
		return session;
	}

	const stored = readSession();
	if (stored && now - stored.lastSeen < SESSION_TIMEOUT_MS) {
		session = { ...stored, lastSeen: now };
		writeSession(session);
		return session;
	}

	session = {
		id: Math.floor(now / 1000),
		count: (stored?.count ?? 0) + 1,
		lastSeen: now,
	};
	isNewSession = true;
	writeSession(session);
	return session;
}

function readSession(): AnalyticsSession | null {
	const raw = readStorage(SESSION_KEY);
	if (!raw) return null;
	try {
		const value = JSON.parse(raw) as Partial<AnalyticsSession>;
		if (typeof value.id !== 'number' || typeof value.count !== 'number' || typeof value.lastSeen !== 'number') return null;
		if (!Number.isFinite(value.id) || !Number.isFinite(value.count) || !Number.isFinite(value.lastSeen)) return null;
		return { id: value.id, count: value.count, lastSeen: value.lastSeen };
	} catch {
		return null;
	}
}

function writeSession(value: AnalyticsSession): void {
	writeStorage(SESSION_KEY, JSON.stringify(value));
}

function readStorage(key: string): string | null {
	try {
		return window.localStorage.getItem(key);
	} catch {
		return null;
	}
}

function writeStorage(key: string, value: string): void {
	try {
		window.localStorage.setItem(key, value);
	} catch {
		// Ignore blocked storage; analytics can continue with memory values.
	}
}

function randomInt(): number {
	const bytes = new Uint32Array(1);
	crypto.getRandomValues(bytes);
	return bytes[0] || Math.floor(Math.random() * 2 ** 32);
}

function pageNonce(): string {
	return `${Date.now()}${Math.floor(Math.random() * 100000)}`;
}
