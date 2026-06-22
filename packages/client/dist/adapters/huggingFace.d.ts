export interface HuggingFaceModelRepositoryOptions {
    readonly repo: string;
    readonly revision?: string;
    readonly manifestPath?: string;
    readonly proxyBase?: string;
}
/** Build a raw Hugging Face resolve URL for any file in a repo. */
export declare function huggingFaceResolveUrl(repo: string, path: string, revision?: string, proxyBase?: string): string;
/** Build the manifest URL for a Hugging Face model repo. */
export declare function huggingFaceManifestUrl(options: HuggingFaceModelRepositoryOptions): string;
export declare function proxyHuggingFaceUrl(url: string, proxyBase: string | undefined): string;
