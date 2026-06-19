import type { PageLoad } from './$types';

// Thin: only pass route param. TanStack Query handles DB + detail in component.
export const load: PageLoad = ({ params }) => {
	return { workId: params.workId };
};
