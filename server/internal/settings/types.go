package settings

type Document struct {
	SourceFetch SourceFetchSettings `json:"sourceFetch"`
	Pricing     PricingSettings     `json:"pricing"`
	Features    FeatureSettings     `json:"features"`
}

type SourceFetchSettings struct {
	// Origins are gateway origins only: scheme + host, no path/query/fragment.
	// Example: https://927251094806098001.discordsays.com
	Origins []string `json:"origins"`
}

type PricingSettings struct {
	Translation TranslationPricing `json:"translation"`
}

type TranslationPricing struct {
	XuPerPage int `json:"xuPerPage"`
}

type FeatureSettings struct {
	Browse      bool `json:"browse"`
	Translation bool `json:"translation"`
}

type PublicDocument struct {
	SourceFetch PublicSourceFetchSettings `json:"sourceFetch"`
	Features    FeatureSettings           `json:"features"`
}

type PublicSourceFetchSettings struct {
	Origins []string `json:"origins"`
}

func Default() Document {
	return Document{
		SourceFetch: SourceFetchSettings{
			Origins: []string{"https://bunle-cdn-ceu.pages.dev"},
		},
		Pricing: PricingSettings{
			Translation: TranslationPricing{XuPerPage: 5},
		},
		Features: FeatureSettings{Browse: true, Translation: true},
	}
}

func Public(doc Document) PublicDocument {
	return PublicDocument{
		SourceFetch: PublicSourceFetchSettings{
			Origins: doc.SourceFetch.Origins,
		},
		Features: doc.Features,
	}
}
