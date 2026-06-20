import { browser } from '$app/environment';

const SESSION_KEY = 'typoon.discordSession.v1';
const OAUTH_STATE_KEY = 'typoon.discordOAuthState.v1';
const STATE_TTL_MS = 10 * 60 * 1000;
const CACHE_STALE_MS = 15 * 60 * 1000;
const DISCORD_ACTIVITY_REDIRECT_URI = 'https://127.0.0.1';
const initialDiscordActivitySearch = browser ? discordActivitySearchFromLocation(window.location) : '';

export class ApiError extends Error {
	readonly status: number;
	readonly detail?: unknown;

	constructor(status: number, message: string, detail?: unknown) {
		super(message);
		this.name = 'ApiError';
		this.status = status;
		this.detail = detail;
	}
}

export class BackendUnavailableError extends Error {
	constructor() {
		super('Discord tạm thời không phản hồi.');
	}
}

export interface SessionUser {
	id: string;
	display_name: string;
	avatar_url: string | null;
	is_admin: boolean;
	is_guild_member?: boolean;
	email?: string | null;
	tier?: { name: string };
	preferred_target_lang?: string | null;
	roles?: string[];
}

export interface PublicSettings {
	sourceFetch: { origins: string[] };
	features: { browse: boolean; translation: boolean };
}

interface DiscordPlan {
	readonly id: string;
	readonly name: string;
	readonly roleIds: readonly string[];
}

interface DiscordSessionCache {
	readonly accessToken: string;
	readonly tokenType: string;
	readonly scope: string;
	readonly expiresAt: number;
	readonly checkedAt: number;
	readonly user: SessionUser;
}

interface OAuthStateCache {
	readonly state: string;
	readonly returnTo: string;
	readonly createdAt: number;
	readonly codeVerifier?: string;
}

interface DiscordUserResponse {
	readonly id: string;
	readonly username: string;
	readonly discriminator?: string;
	readonly global_name?: string | null;
	readonly avatar?: string | null;
	readonly email?: string | null;
}

interface DiscordGuildMemberResponse {
	readonly roles?: string[];
}

interface DiscordTokenResponse {
	readonly access_token?: string;
	readonly token_type?: string;
	readonly expires_in?: number;
	readonly scope?: string;
}

export const discordEntitlementConfig = {
	clientId: env('VITE_DISCORD_CLIENT_ID'),
	guildId: env('VITE_DISCORD_GUILD_ID'),
	proxyBase: env('VITE_DISCORD_PROXY_BASE') || '/discord',
	scopes: ['identify', 'guilds.members.read'] as const,
	cacheGraceHours: numberEnv('VITE_DISCORD_CACHE_GRACE_HOURS', 168),
	plans: [
		{ id: 'vip', name: 'VIP', roleIds: csvEnv('VITE_DISCORD_VIP_ROLE_IDS') },
		{ id: 'pro', name: 'Pro', roleIds: csvEnv('VITE_DISCORD_PRO_ROLE_IDS') },
	] satisfies DiscordPlan[],
};

export function missingDiscordConfig(): string[] {
	const missing: string[] = [];
	if (!discordEntitlementConfig.clientId) missing.push('VITE_DISCORD_CLIENT_ID');
	if (!discordEntitlementConfig.guildId) missing.push('VITE_DISCORD_GUILD_ID');
	return missing;
}

export const isDiscordActivity = !!initialDiscordActivitySearch;

export const authApi = {
	getSession,
	logout,
	getSettings: async (): Promise<PublicSettings> => ({
		sourceFetch: { origins: [] },
		features: { browse: true, translation: true },
	}),
};

export function setDaToken(_token: string | null): void {
	// Legacy Discord Activity hook. FE-only auth stores the Discord user token in localStorage.
}

export function loginUrl(returnTo = '/'): string {
	if (!browser) return '/login';
	const missing = missingDiscordConfig();
	if (missing.length > 0) {
		return `/login?error=${encodeURIComponent(`Thiếu cấu hình Discord khi build: ${missing.join(', ')}.`)}`;
	}
	const state = createOAuthState(returnTo);
	const params = new URLSearchParams({
		client_id: discordEntitlementConfig.clientId,
		response_type: 'token',
		redirect_uri: callbackUrl(),
		scope: discordEntitlementConfig.scopes.join(' '),
		state,
		prompt: 'consent',
	});
	return `https://discord.com/oauth2/authorize?${params.toString()}`;
}

export async function startDiscordLogin(returnTo = '/'): Promise<void> {
	if (!browser) return;
	const missing = missingDiscordConfig();
	if (missing.length > 0) {
		window.location.href = `/login?error=${encodeURIComponent(`Thiếu cấu hình Discord khi build: ${missing.join(', ')}.`)}`;
		return;
	}
	if (!crypto.subtle) {
		window.location.href = loginUrl(returnTo);
		return;
	}

	const codeVerifier = randomCodeVerifier();
	const state = createOAuthState(returnTo, codeVerifier);
	const params = new URLSearchParams({
		client_id: discordEntitlementConfig.clientId,
		response_type: 'code',
		redirect_uri: callbackUrl(),
		scope: discordEntitlementConfig.scopes.join(' '),
		state,
		prompt: 'consent',
		code_challenge: await pkceChallenge(codeVerifier),
		code_challenge_method: 'S256',
	});
	window.location.href = `https://discord.com/oauth2/authorize?${params.toString()}`;
}

export function safeReturnTo(value: unknown): string {
	if (typeof value !== 'string') return '/';
	if (!value.startsWith('/') || value.startsWith('//')) return '/';
	if (value.startsWith('/login') || value.startsWith('/auth/callback')) return '/';
	return value;
}

export async function discordActivityLogin(): Promise<void> {
	if (!browser) return;
	if (!isDiscordActivity) throw new Error('Trang này không chạy trong Discord Activity.');
	const missing = missingDiscordConfig();
	if (missing.length > 0) throw new Error(`Thiếu cấu hình Discord khi build: ${missing.join(', ')}.`);
	if (!crypto.subtle) throw new Error('Trình duyệt không hỗ trợ Discord Activity PKCE.');

	const activitySearch = initialDiscordActivitySearch || discordActivitySearchFromLocation(window.location);
	if (!activitySearch) throw new Error('Discord Activity thiếu frame_id.');

	const { DiscordSDK } = await import('@discord/embedded-app-sdk');
	class DiscordActivitySDK extends DiscordSDK {
		override _getSearch(): string { return activitySearch; }
	}
	const discordSdk = new DiscordActivitySDK(discordEntitlementConfig.clientId);
	await discordSdk.ready();

	const codeVerifier = randomCodeVerifier();
	const { code } = await discordSdk.commands.authorize({
		client_id: discordEntitlementConfig.clientId,
		response_type: 'code',
		prompt: 'none',
		scope: ['identify', 'guilds.members.read'],
		code_challenge: await pkceChallenge(codeVerifier),
		code_challenge_method: 'S256',
	});

	const token = await exchangeDiscordCode(code, codeVerifier, DISCORD_ACTIVITY_REDIRECT_URI);
	const accessToken = token.access_token ?? '';
	const expiresIn = Number(token.expires_in ?? '0');
	if (!accessToken || !Number.isFinite(expiresIn) || expiresIn <= 0) {
		throw new Error('Discord không trả về access token hợp lệ.');
	}

	await discordSdk.commands.authenticate({ access_token: accessToken });
	await refreshDiscordSession({
		accessToken,
		tokenType: token.token_type ?? 'Bearer',
		scope: token.scope ?? '',
		expiresAt: Date.now() + expiresIn * 1000,
	});
}

export async function exchangeDiscordCallback(fragment: URLSearchParams, query = new URLSearchParams()): Promise<string> {
	const error = query.get('error') ?? fragment.get('error');
	if (error) throw new Error(query.get('error_description') ?? fragment.get('error_description') ?? error);

	if (query.get('code') || query.get('state')) {
		return exchangeDiscordCodeCallback(query);
	}

	const accessToken = fragment.get('access_token');
	const tokenType = fragment.get('token_type') ?? 'Bearer';
	const expiresIn = Number(fragment.get('expires_in') ?? '0');
	const scope = fragment.get('scope') ?? '';
	const state = fragment.get('state') ?? '';
	if (!accessToken || !state || !Number.isFinite(expiresIn) || expiresIn <= 0) {
		throw new Error('Callback Discord thiếu access token.');
	}

	const returnTo = consumeOAuthState(state);
	await refreshDiscordSession({
		accessToken,
		tokenType,
		scope,
		expiresAt: Date.now() + expiresIn * 1000,
	});
	return returnTo;
}

async function exchangeDiscordCodeCallback(query: URLSearchParams): Promise<string> {
	const code = query.get('code') ?? '';
	const state = query.get('state') ?? '';
	if (!code || !state) throw new Error('Callback Discord thiếu authorization code.');

	const cached = consumeOAuthStateCache(state);
	if (!cached.codeVerifier) throw new Error('Phiên đăng nhập Discord cũ không có PKCE verifier. Hãy đăng nhập lại.');

	const token = await exchangeDiscordCode(code, cached.codeVerifier, callbackUrl());
	const accessToken = token.access_token ?? '';
	const expiresIn = Number(token.expires_in ?? '0');
	if (!accessToken || !Number.isFinite(expiresIn) || expiresIn <= 0) {
		throw new Error('Discord không trả về access token hợp lệ.');
	}

	await refreshDiscordSession({
		accessToken,
		tokenType: token.token_type ?? 'Bearer',
		scope: token.scope ?? '',
		expiresAt: Date.now() + expiresIn * 1000,
	});
	return cached.returnTo;
}

async function getSession(): Promise<SessionUser> {
	const cached = readSession();
	if (!cached) throw new ApiError(401, 'Unauthorized');

	const now = Date.now();
	const graceMs = discordEntitlementConfig.cacheGraceHours * 60 * 60 * 1000;
	const inGrace = now - cached.checkedAt <= graceMs;
	if (cached.user.is_guild_member === undefined && now < cached.expiresAt) return refreshDiscordSession(cached);
	if (now - cached.checkedAt < CACHE_STALE_MS) return cached.user;

	if (now < cached.expiresAt) {
		try {
			return await refreshDiscordSession(cached);
		} catch (err) {
			if (inGrace) return cached.user;
			throw err;
		}
	}

	if (inGrace) return cached.user;
	clearSession();
	throw new ApiError(401, 'Unauthorized');
}

async function logout(): Promise<{ ok: boolean }> {
	clearSession();
	return { ok: true };
}

async function refreshDiscordSession(token: Pick<DiscordSessionCache, 'accessToken' | 'tokenType' | 'scope' | 'expiresAt'>): Promise<SessionUser> {
	const discordUser = await discordFetch<DiscordUserResponse>('/me', token.accessToken);
	const membership = await fetchDiscordMembership(token.accessToken);
	const plan = resolvePlan(membership.roles);
	const user: SessionUser = {
		id: discordUser.id,
		display_name: discordUser.global_name || discordUser.username,
		avatar_url: avatarUrl(discordUser),
		is_admin: false,
		is_guild_member: membership.isGuildMember,
		email: discordUser.email ?? null,
		tier: plan ? { name: plan.name } : undefined,
		preferred_target_lang: null,
		roles: membership.roles,
	};
	writeSession({ ...token, checkedAt: Date.now(), user });
	return user;
}

async function fetchDiscordMembership(accessToken: string): Promise<{ roles: string[]; isGuildMember: boolean }> {
	if (!discordEntitlementConfig.guildId) return { roles: [], isGuildMember: true };
	try {
		return {
			roles: (await discordFetch<DiscordGuildMemberResponse>(`/member?guild_id=${encodeURIComponent(discordEntitlementConfig.guildId)}`, accessToken)).roles ?? [],
			isGuildMember: true,
		};
	} catch (err) {
		if (err instanceof ApiError && err.status === 404) return { roles: [], isGuildMember: false };
		throw err;
	}
}

async function discordFetch<T>(path: string, accessToken: string): Promise<T> {
	let res: Response;
	try {
		res = await fetch(`${discordEntitlementConfig.proxyBase}${path}`, { headers: { Authorization: `Bearer ${accessToken}` } });
	} catch {
		throw new BackendUnavailableError();
	}
	if (res.status === 401 || res.status === 403) throw new ApiError(401, 'Discord authorization expired');
	if (!res.ok) throw new ApiError(res.status, `Discord API lỗi ${res.status}`);
	return res.json() as Promise<T>;
}

async function exchangeDiscordCode(code: string, codeVerifier: string, redirectURI: string): Promise<DiscordTokenResponse> {
	let res: Response;
	try {
		res = await fetch(`${discordEntitlementConfig.proxyBase}/token`, {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({
				client_id: discordEntitlementConfig.clientId,
				code,
				code_verifier: codeVerifier,
				redirect_uri: redirectURI,
			}),
		});
	} catch {
		throw new BackendUnavailableError();
	}
	const detail = await res.json().catch(() => null) as unknown;
	if (!res.ok) {
		if (discordErrorCode(detail) === 'invalid_client') {
			throw new Error('Discord app chưa bật Public Client cho PKCE. Bật Public Client trong Developer Portal rồi đăng nhập lại.');
		}
		throw new ApiError(res.status, `Discord token exchange lỗi ${res.status}`, detail);
	}
	return detail as DiscordTokenResponse;
}

function resolvePlan(roles: readonly string[]): DiscordPlan | null {
	const roleSet = new Set(roles);
	return discordEntitlementConfig.plans.find((plan) => plan.roleIds.some((roleId) => roleSet.has(roleId))) ?? null;
}

function callbackUrl(): string {
	return `${window.location.origin}/auth/callback`;
}

function discordActivitySearchFromLocation(location: Location): string {
	if (!location.hostname.endsWith('.discordsays.com')) return '';
	const direct = searchWithFrameId(location.search);
	if (direct) return direct;

	const redirect = new URLSearchParams(location.search).get('redirect') ?? '';
	if (!redirect) return '';
	try {
		return searchWithFrameId(new URL(redirect, location.origin).search);
	} catch {
		return '';
	}
}

function searchWithFrameId(search: string): string {
	return new URLSearchParams(search).has('frame_id') ? search : '';
}

function createOAuthState(returnTo: string, codeVerifier?: string): string {
	const state = randomState();
	writeJson(OAUTH_STATE_KEY, { state, returnTo: safeReturnTo(returnTo), createdAt: Date.now(), codeVerifier } satisfies OAuthStateCache);
	return state;
}

function consumeOAuthState(state: string): string {
	return consumeOAuthStateCache(state).returnTo;
}

function consumeOAuthStateCache(state: string): OAuthStateCache {
	const cached = readJson<OAuthStateCache>(OAUTH_STATE_KEY);
	localStorage.removeItem(OAUTH_STATE_KEY);
	if (!cached || cached.state !== state || Date.now() - cached.createdAt > STATE_TTL_MS) {
		throw new Error('Phiên đăng nhập Discord đã hết hạn.');
	}
	return cached;
}

function readSession(): DiscordSessionCache | null {
	return readJson<DiscordSessionCache>(SESSION_KEY);
}

function writeSession(value: DiscordSessionCache): void {
	writeJson(SESSION_KEY, value);
}

function clearSession(): void {
	if (!browser) return;
	localStorage.removeItem(SESSION_KEY);
	localStorage.removeItem(OAUTH_STATE_KEY);
}

function readJson<T>(key: string): T | null {
	if (!browser) return null;
	try {
		const raw = localStorage.getItem(key);
		return raw ? JSON.parse(raw) as T : null;
	} catch {
		return null;
	}
}

function writeJson(key: string, value: unknown): void {
	if (!browser) return;
	localStorage.setItem(key, JSON.stringify(value));
}

function avatarUrl(user: DiscordUserResponse): string | null {
	if (!user.avatar) return null;
	const ext = user.avatar.startsWith('a_') ? 'gif' : 'png';
	return `https://cdn.discordapp.com/avatars/${user.id}/${user.avatar}.${ext}?size=128`;
}

function randomState(): string {
	const bytes = new Uint8Array(16);
	crypto.getRandomValues(bytes);
	return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

function randomCodeVerifier(): string {
	const bytes = new Uint8Array(32);
	crypto.getRandomValues(bytes);
	return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function pkceChallenge(codeVerifier: string): Promise<string> {
	const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(codeVerifier));
	return base64Url(new Uint8Array(digest));
}

function base64Url(bytes: Uint8Array): string {
	let raw = '';
	bytes.forEach((byte) => { raw += String.fromCharCode(byte); });
	return btoa(raw).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function discordErrorCode(detail: unknown): string {
	if (!detail || typeof detail !== 'object') return '';
	const error = (detail as { error?: unknown }).error;
	return typeof error === 'string' ? error : '';
}

function env(key: string): string {
	return String(import.meta.env[key] ?? '').trim();
}

function csvEnv(key: string): string[] {
	return env(key).split(',').map((value) => value.trim()).filter(Boolean);
}

function numberEnv(key: string, fallback: number): number {
	const value = Number(env(key));
	return Number.isFinite(value) && value > 0 ? value : fallback;
}
