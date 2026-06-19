import type { TranslatedUnit } from '../../domain/translation';
import type { Translator } from '../translator';
export interface DeepLTranslateWebOptions {
    readonly endpoint?: string;
    readonly websocketOrigin?: string;
    readonly requestTimeoutMs?: number;
    readonly maxBatchChars?: number;
    readonly credentials?: RequestCredentials;
    readonly maxSessions?: number;
}
export declare class DeepLTranslateWeb implements Translator {
    readonly name = "deepl-translate-web";
    private readonly endpoint;
    private readonly websocketOrigin;
    private readonly requestTimeoutMs;
    private readonly maxBatchChars;
    private readonly credentials;
    private readonly limiter;
    private sessions;
    private readonly busy;
    constructor(options?: DeepLTranslateWebOptions);
    translateUnits({ units, sourceLang, targetLang, signal }: Parameters<Translator['translateUnits']>[0]): Promise<readonly TranslatedUnit[]>;
    close(): Promise<void>;
    private acquireSession;
}
