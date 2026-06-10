package translation

import "fmt"

const keyAlphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

func assignPromptKey(sessionID string, pageIndex, order int, used map[string]bool) string {
	salt := 0

	for {
		key := makePromptKey(sessionID, pageIndex, order, salt)
		if !used[key] {
			used[key] = true

			return key
		}

		salt++
	}
}

func makePromptKey(sessionID string, pageIndex, order, salt int) string {
	n := fnv1a(sessionID, pageIndex, order, salt)
	out := make([]byte, 7)

	for i := range out {
		out[i] = keyAlphabet[n%uint32(len(keyAlphabet))]
		n = (n / uint32(len(keyAlphabet))) ^ (n << 7)
	}

	return string(out)
}

func fnv1a(sessionID string, pageIndex, order, salt int) uint32 {
	s := fmt.Sprintf("%s:%d:%d:%d", sessionID, pageIndex, order, salt)
	hash := uint32(0x811c9dc5)

	for i := range len(s) {
		hash ^= uint32(s[i])
		hash *= 0x01000193
	}

	return hash
}
