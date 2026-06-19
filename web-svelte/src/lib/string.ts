export function stripHtml(value: string): string {
	return value.replace(/<[^>]+>/g, '').replace(/\s+\n/g, '\n').trim();
}
