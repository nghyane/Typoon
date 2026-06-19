export type TextScript = 'latin' | 'hangul' | 'kana' | 'han' | 'mixed-cjk' | 'numeric-symbol' | 'mixed';
export declare function classifyTextScript(text: string): TextScript;
export declare function canUseVerticalTypesetting(text: string): boolean;
export declare function canBreakTokenPerCharacter(text: string): boolean;
