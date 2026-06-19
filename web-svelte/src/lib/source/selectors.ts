const FALLBACK_SEP = /\s*\|\|\s*/;

function isEmpty(value: unknown): boolean {
	if (value == null) return true;
	if (typeof value === 'string') return value.trim().length === 0;
	if (Array.isArray(value)) return value.length === 0;
	return false;
}

function queryScriptSpecial(root: Element | Document, selector: string): string | null {
	const jsonMatch = selector.match(/^script:json\(([^)]+)\)$/);
	const regexMatch = selector.match(/^script:match\((.+)\)$/);
	if (!jsonMatch && !regexMatch) return null;

	const scripts = Array.from(root.querySelectorAll('script'));

	if (jsonMatch) {
		const varName = jsonMatch[1]!.trim();
		for (const script of scripts) {
			const text = script.textContent ?? '';
			if (!text.includes(varName)) continue;
			const patterns = [
				new RegExp(String.raw`\b${varName}\s*=\s*(\{[\s\S]*?\})\s*[;,\n]`),
				new RegExp(String.raw`\b${varName}\s*=\s*(\[[\s\S]*?\])\s*[;,\n]`),
				new RegExp(String.raw`\b${varName}\s*=\s*(?:\$\.parseJSON|JSON\.parse)\s*\(\s*['"](.+?)['"]\s*\)`),
			];
			for (const re of patterns) {
				const match = re.exec(text);
				if (match?.[1]) {
					try {
						JSON.parse(match[1]);
						return match[1];
					} catch { /* try next pattern */ }
				}
			}
		}
		return null;
	}

	const userRegex = regexMatch![1]!.trim();
	let re: RegExp;
	try { re = new RegExp(userRegex); } catch { return null; }

	for (const script of scripts) {
		const match = re.exec(script.textContent ?? '');
		if (match?.[1] != null) return match[1];
	}
	return null;
}

export function queryHtmlOne(root: Element | Document, selector: string): string | null {
	for (const sel of selector.split(FALLBACK_SEP)) {
		const value = queryHtmlOneSingle(root, sel);
		if (!isEmpty(value)) return value;
	}
	return null;
}

function queryHtmlOneSingle(root: Element | Document, selector: string): string | null {
	if (selector.startsWith('script:')) return queryScriptSpecial(root, selector);

	const { sel, attr } = splitAttr(selector);
	const el = sel ? root.querySelector(sel) : (root as Element);
	if (!el) return null;
	if (attr) return el.getAttribute(attr);
	return cleanText(el.textContent);
}

export function queryHtmlAll(root: Element | Document, selector: string): Element[] {
	return Array.from(root.querySelectorAll(selector));
}

export function queryJsonOne(root: unknown, selector: string): unknown {
	for (const sel of selector.split(FALLBACK_SEP)) {
		const value = queryJsonOneSingle(root, sel);
		if (!isEmpty(value)) return value;
	}
	return null;
}

function queryJsonOneSingle(root: unknown, selector: string): unknown {
	const trimmed = selector.trim();
	const at = trimmed.indexOf('@');
	if (at < 0) return evalJsonPath(root, trimmed);

	const anchorRaw = trimmed.slice(0, at).trim();
	const anchor = anchorRaw.length === 0 ? root : evalJsonPath(root, anchorRaw);
	if (anchor == null) return null;

	const attrExpr = trimmed.slice(at + 1).trim();
	if (!attrExpr) return anchor;
	return resolveAttrExpr(anchor, attrExpr);
}

function resolveAttrExpr(anchor: unknown, expr: string): unknown {
	if (Array.isArray(anchor)) {
		for (const item of anchor) {
			const result = resolveAttrExpr(item, expr);
			if (!isEmpty(result)) return result;
		}
		return null;
	}

	const at = expr.indexOf('@');
	if (at < 0) return evalJsonPath(anchor, '$.' + expr);

	const left = expr.slice(0, at).trim();
	const right = expr.slice(at + 1).trim();
	const value = evalJsonPath(anchor, '$.' + left);
	if (Array.isArray(value)) {
		for (const item of value) {
			const result = resolveAttrExpr(item, right);
			if (!isEmpty(result)) return result;
		}
		return null;
	}
	return resolveAttrExpr(value, right);
}

export function queryJsonAll(root: unknown, selector: string): unknown[] {
	for (const sel of selector.split(FALLBACK_SEP)) {
		const value = queryJsonAllSingle(root, sel);
		if (value.length > 0) return value;
	}
	return [];
}

function queryJsonAllSingle(root: unknown, selector: string): unknown[] {
	const trimmed = selector.trim();
	const at = trimmed.indexOf('@');
	if (at < 0) {
		const sel = trimmed.endsWith('[*]') || trimmed.endsWith('.*') ? trimmed : trimmed + '[*]';
		const value = evalJsonPath(root, sel);
		return Array.isArray(value) ? value.filter((item) => item != null) : [];
	}

	const anchorRaw = trimmed.slice(0, at).trim();
	const expr = trimmed.slice(at + 1).trim();
	const anchor = anchorRaw.length === 0 ? root : evalJsonPath(root, anchorRaw);
	if (!Array.isArray(anchor)) return [];
	return anchor.map((item) => resolveAttrExpr(item, expr)).filter((item) => item != null);
}

function splitAttr(value: string): { sel: string; attr: string | null } {
	const trimmed = value.trim();
	const at = trimmed.indexOf('@');
	if (at < 0) return { sel: trimmed, attr: null };
	return {
		sel: trimmed.slice(0, at).trim(),
		attr: trimmed.slice(at + 1).trim() || null,
	};
}

function cleanText(value: string | null | undefined): string | null {
	if (!value) return null;
	const out = value.replace(/\s+/g, ' ').trim();
	return out.length > 0 ? out : null;
}

function evalJsonPath(root: unknown, path: string): unknown {
	if (!path.startsWith('$')) return null;
	let current: unknown = root;
	let i = 1;
	while (i < path.length && current != null) {
		const char = path[i];
		if (char === '.') {
			if (path[i + 1] === '*') {
				if (current && typeof current === 'object' && !Array.isArray(current)) current = Object.values(current);
				else if (!Array.isArray(current)) return null;

				const rest = path.slice(i + 2);
				if (!rest) return firstNonEmpty(current as unknown[]);
				const mapped = (current as unknown[]).map((item) => evalJsonPath(item, `$${rest}`));
				if (rest.startsWith('[*]') || rest.startsWith('.*')) return (mapped as unknown[][]).flat();
				return mapped;
			}

			const match = /^\.([A-Za-z_][\w-]*)/.exec(path.slice(i));
			if (!match) return null;
			current = (current as Record<string, unknown>)[match[1]!];
			i += match[0].length;
		} else if (char === '[') {
			const close = path.indexOf(']', i);
			if (close < 0) return null;
			const index = path.slice(i + 1, close).trim();

			if (index === '*') {
				if (!Array.isArray(current)) {
					if (current && typeof current === 'object') current = Object.values(current);
					else return [];
				}
				const rest = path.slice(close + 1);
				if (!rest) return current;
				return (current as unknown[]).map((item) => evalJsonPath(item, `$${rest}`));
			}

			if (index.startsWith('?')) {
				const eq = index.indexOf('=', 1);
				if (eq < 0) return null;
				const key = index.slice(1, eq).trim();
				let expected = index.slice(eq + 1).trim();
				if ((expected.startsWith("'") && expected.endsWith("'"))
					|| (expected.startsWith('"') && expected.endsWith('"'))) {
					expected = expected.slice(1, -1);
				}
				if (!Array.isArray(current)) return null;
				const kept = (current as unknown[]).filter((item) => {
					if (item == null || typeof item !== 'object') return false;
					const value = (item as Record<string, unknown>)[key];
					return value != null && String(value) === expected;
				});
				const rest = path.slice(close + 1);
				if (!rest) return kept.length > 0 ? kept[0] : null;
				if (rest.startsWith('[*]') || rest.startsWith('.*')) {
					return kept.map((item) => evalJsonPath(item, `$${rest}`));
				}
				if (kept.length === 0) return null;
				return evalJsonPath(kept[0], `$${rest}`);
			}

			const n = Number(index);
			if (!Array.isArray(current) || !Number.isInteger(n)) return null;
			current = current[n];
			i = close + 1;
		} else {
			return null;
		}
	}
	return current;
}

function firstNonEmpty(values: unknown[]): unknown {
	for (const value of values) {
		if (value == null) continue;
		if (typeof value === 'string' && value.trim().length === 0) continue;
		if (Array.isArray(value) && value.length === 0) continue;
		return value;
	}
	return null;
}
