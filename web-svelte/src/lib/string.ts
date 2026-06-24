export function stripHtml(value: string): string {
	return value.replace(/<[^>]+>/g, '').replace(/\s+\n/g, '\n').trim();
}

const STATUS_MAP: Record<string, string> = {
	ongoing: 'Đang tiến hành', 'on going': 'Đang tiến hành', 'on-going': 'Đang tiến hành',
	releasing: 'Đang tiến hành', publishing: 'Đang tiến hành',
	'đang cập nhật': 'Đang tiến hành', 'đang tiến hành': 'Đang tiến hành',
	completed: 'Hoàn thành', complete: 'Hoàn thành', finished: 'Hoàn thành',
	'hoàn thành': 'Hoàn thành', 'hoàn tất': 'Hoàn thành',
	hiatus: 'Tạm ngưng', 'on hiatus': 'Tạm ngưng', 'tạm ngưng': 'Tạm ngưng',
	cancelled: 'Đã huỷ', canceled: 'Đã huỷ', dropped: 'Đã huỷ',
};

export function normalizeStatus(s: string | null | undefined): string | null {
	if (!s) return null;
	return STATUS_MAP[s.toLowerCase().trim()] ?? s;
}

export function timeAgo(value: string): string {
	const diff = Math.max(0, Date.now() - Date.parse(value));
	const minutes = Math.floor(diff / 60_000);
	if (minutes < 1) return 'vừa xong';
	if (minutes < 60) return `${minutes} phút trước`;
	const hours = Math.floor(minutes / 60);
	if (hours < 24) return `${hours} giờ trước`;
	const days = Math.floor(hours / 24);
	if (days < 30) return `${days} ngày trước`;
	return value.slice(0, 10);
}
