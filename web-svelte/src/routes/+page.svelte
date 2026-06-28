<script lang="ts">
  import { onMount } from 'svelte';
  import { ArrowRight, Check, ChevronRight, Smartphone, Globe, Puzzle, Heart, Zap, BadgeCheck, Palette, Sparkles } from 'lucide-svelte';
  import { BRAND } from '$lib/brand';
  import { session } from '$lib/auth/session.svelte';

  // Canonical production origin — used for SEO tags. Keep in sync with the deploy.
  const ORIGIN = 'https://hoimetruyen.net';
  const TITLE = `${BRAND.name} — Đọc truyện tranh kèm dịch tiếng Việt tức thì`;
  const DESCRIPTION =
    'Mở truyện tranh tiếng Nhật, Hàn, Trung và đọc ngay bằng tiếng Việt. Bản dịch hiện thẳng lên trang — miễn phí, không phải chờ ai, mở là đọc được.';
  const OG_IMAGE = `${ORIGIN}/pwa/icon-512.png`;
  // TODO: trỏ tới trang ủng hộ thật (Ko-fi / MoMo / kênh Discord) khi sẵn sàng.
  const SUPPORT_HREF = '#';

  // `/` always shows this marketing page — including for logged-in users — so the
  // landing stays viewable instead of bouncing authenticated visitors to /home.
  //
  // The CTAs, however, should skip the login screen once we already have a
  // session: logged-out visitors (and crawlers, who get the prerendered HTML)
  // go to /login?redirect=/home; logged-in visitors go straight to /home. The
  // session check is client-only, so the indexable HTML keeps the login href.
  let ctaHref = $state('/login?redirect=%2Fhome');
  onMount(() => { void session.load(); });
  $effect(() => {
    if (session.state.status === 'authenticated') ctaHref = '/home';
  });

  // Where it runs. Ext is on the roadmap — `ready: false` renders a "Sắp có" badge
  // so the slot exists now and lights up when the extension ships.
  const platforms = [
    { icon: Globe, name: 'Trình duyệt', body: 'Chrome, Edge, Safari — không cài gì.', status: 'Có sẵn', ready: true },
    { icon: Smartphone, name: 'Ứng dụng (PWA)', body: 'Cài về máy, mở nhanh như app.', status: 'Có sẵn', ready: true },
    { icon: Puzzle, name: 'Tiện ích trình duyệt', body: 'Đọc ngay trên trang nguồn.', status: 'Sắp có', ready: false },
  ];

  // Supporter perks — deliberately NONE of these unlock content. Paying buys speed,
  // status, and cosmetics (things we actually own); basic translation stays free so
  // the tool remains a non-commercial reader, not a paid piracy service.
  const supporterPerks = [
    { icon: Zap, title: 'Chất lượng cao, không giới hạn', body: 'Model dịch lớn hơn — nét & chính xác hơn, đọc bao nhiêu tuỳ thích.' },
    { icon: BadgeCheck, title: 'Huy hiệu & role Discord', body: 'Danh phận riêng trong cộng đồng người ủng hộ.' },
    { icon: Palette, title: 'Tuỳ chỉnh font & giao diện', body: 'Font bản dịch và theme riêng theo ý bạn.' },
    { icon: Sparkles, title: 'Tính năng mới sớm nhất', body: 'Dùng trước mỗi khi có thứ gì mới ra lò.' },
  ];

  // The reader-demo overlay is the app's *real* placement output for one licensed
  // page (provided by the site owner at static/demo/reader-sample.webp). Coords are
  // % of the page box (aspect 1125/1600); font sizes in cqw, exactly as the reader emits.
  const overlayLines = [
    { id: 'r4', left: 80.2667, top: 9.875, w: 9.24444, h: 11.5625, px: 0.977778, py: 0.622222, fs: 1.6, lh: 2.62032, text: 'CÚ ĐÁ\nMÀ CẬU\nĐÃ DẠY\nTÔI…' },
    { id: 'r11', left: 75.8667, top: 26.7812, w: 12.6222, h: 13.3125, px: 1.06667, py: 0.711111, fs: 1.77778, lh: 2.91147, text: 'ĐÓ CHÍNH\nLÀ ĐỘNG\nTÁC MÀ TÔI\nĐÃ MONG\nMỘT THỜI\nLÂU RỒI.' },
    { id: 'r3', left: 23.6444, top: 50.8125, w: 9.68889, h: 8.6875, px: 0.977778, py: 0.622222, fs: 1.6, lh: 2.62032, text: 'Ồ? KHÔNG\nĐƯỢC\nQUAY LẠI!' },
    { id: 'r9', left: 9.24444, top: 50.875, w: 13.6889, h: 11.375, px: 0.977778, py: 0.622222, fs: 1.6, lh: 2.57778, text: 'TÔI SẼ\nCHỨNG TỎ CHO\nBẠN THẤY TÔI\nCÓ BAO NHIÊU\nSỨC MẠNH!' },
    { id: 'r10', left: 70.1333, top: 52.5, w: 16.3556, h: 9.1875, px: 0.8, py: 0.622222, fs: 1.33333, lh: 2.1836, text: 'ĐƯỢC RỒI, CHỈ HÔM\nNAY THÔI, TÔI SẼ ĐẤU\nVỚI BẠN BAO NHIÊU\nLẦN CŨNG ĐƯỢC.' },
    { id: 'r7', left: 11.7333, top: 87.5312, w: 10.2222, h: 6.125, px: 0.977778, py: 0.622222, fs: 1.51111, lh: 2.47475, text: 'Ồ, BẠN\nĐÃ TỈNH\nRỒI SAO?' },
  ];
  // White mask rects (source units, viewBox 1125×1600) that cover the original text.
  const maskRects = [
    [925, 156, 58, 29], [905, 195, 99, 30], [924, 234, 62, 32], [903, 276, 104, 30], [917, 315, 49, 30], [957, 315, 32, 30],
    [894, 432, 63, 36], [862, 475, 63, 35], [920, 476, 62, 32], [858, 517, 58, 37], [910, 517, 81, 35], [858, 560, 129, 35], [886, 603, 66, 35], [939, 606, 17, 30],
    [265, 813, 53, 56], [304, 816, 29, 50], [322, 812, 53, 56], [278, 858, 83, 37], [267, 898, 96, 55], [341, 901, 28, 47],
    [115, 814, 61, 59], [159, 812, 90, 60], [119, 872, 64, 39], [173, 872, 71, 38], [103, 900, 90, 55], [176, 900, 83, 54], [122, 955, 59, 40], [168, 955, 69, 39], [213, 955, 35, 38],
    [801, 838, 96, 29], [892, 839, 60, 25], [946, 839, 12, 26], [787, 869, 65, 29], [848, 869, 52, 28], [896, 870, 78, 25], [959, 870, 15, 26],
    [801, 900, 48, 25], [848, 900, 62, 24], [908, 900, 55, 24], [805, 934, 52, 23], [855, 934, 37, 23], [889, 934, 69, 23],
    [807, 964, 34, 24], [842, 964, 50, 23], [891, 964, 63, 23], [942, 964, 12, 24],
    [161, 1401, 48, 30], [198, 1403, 18, 27], [141, 1434, 98, 30], [135, 1470, 92, 28], [213, 1471, 30, 25],
  ];

  const faqs = [
    {
      q: 'Hội Mê Truyện có lưu trữ truyện không?',
      a: 'Không. Đây là trình đọc kèm công cụ dịch. Nội dung đến từ các nguồn bên thứ ba mà chính bạn chọn và bật — chúng tôi không lưu trữ hay phân phối nội dung có bản quyền.',
    },
    {
      q: 'Bản dịch hoạt động như thế nào?',
      a: 'AI tự đọc chữ trong các khung thoại, dịch sang tiếng Việt rồi hiện ngay lên đúng vị trí — bạn không phải làm gì thêm.',
    },
    {
      q: 'Dùng có mất phí không?',
      a: 'Đọc và dịch cơ bản hoàn toàn miễn phí — đăng nhập bằng Discord là bắt đầu được ngay. Ủng hộ là tự nguyện: chỉ mở thêm vài đặc quyền như dịch nhanh hơn, role Discord hay tuỳ chỉnh giao diện, và giúp trả phí máy chủ. Bạn không cần trả gì để đọc thoải mái.',
    },
    {
      q: 'Truyện tôi đọc có riêng tư không?',
      a: 'Có. Việc dịch diễn ra ngay trên thiết bị của bạn và thư viện được lưu trong trình duyệt — không gửi truyện bạn đọc lên máy chủ nào.',
    },
    {
      q: 'Cần thiết bị gì?',
      a: 'Một trình duyệt hiện đại trên máy tính hoặc điện thoại. Có thể cài như ứng dụng (PWA) để mở nhanh hơn.',
    },
  ];

  // JSON-LD — WebApplication describes the tool; FAQPage mirrors the visible Q&A.
  const jsonLd = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'WebApplication',
        name: BRAND.name,
        url: `${ORIGIN}/`,
        applicationCategory: 'BookApplication',
        operatingSystem: 'Web',
        browserRequirements: 'Requires a modern web browser with JavaScript enabled.',
        description: DESCRIPTION,
        inLanguage: 'vi',
        offers: { '@type': 'Offer', price: '0', priceCurrency: 'VND' },
      },
      {
        '@type': 'FAQPage',
        mainEntity: faqs.map((item) => ({
          '@type': 'Question',
          name: item.q,
          acceptedAnswer: { '@type': 'Answer', text: item.a },
        })),
      },
    ],
  };
</script>

<svelte:head>
  <title>{TITLE}</title>
  <meta name="description" content={DESCRIPTION} />
  <link rel="canonical" href={`${ORIGIN}/`} />
  <meta name="robots" content="index, follow" />
  <meta property="og:type" content="website" />
  <meta property="og:site_name" content={BRAND.name} />
  <meta property="og:title" content={TITLE} />
  <meta property="og:description" content={DESCRIPTION} />
  <meta property="og:url" content={`${ORIGIN}/`} />
  <meta property="og:image" content={OG_IMAGE} />
  <meta property="og:locale" content="vi_VN" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content={TITLE} />
  <meta name="twitter:description" content={DESCRIPTION} />
  <meta name="twitter:image" content={OG_IMAGE} />
  {@html `<script type="application/ld+json">${JSON.stringify(jsonLd)}</script>`}
</svelte:head>

<div class="min-h-dvh bg-bg text-text">
  <!-- Nav — solid, no blur, no border. The page scrolls cleanly underneath. -->
  <header class="sticky top-0 z-30 bg-bg">
    <div class="mx-auto flex h-14 max-w-5xl items-center justify-between px-4 sm:px-6">
      <a href="/" class="flex items-center gap-2.5">
        <span class="grid size-7 place-items-center overflow-hidden rounded-sm bg-accent text-[10px] font-bold leading-none text-accent-fg">
          {#if BRAND.logoUrl}<img src={BRAND.logoUrl} alt="" class="size-full object-cover" />{:else}{BRAND.monogram}{/if}
        </span>
        <span class="text-sm font-semibold tracking-tight">{BRAND.name}</span>
      </a>
      <nav class="flex items-center gap-1">
        <a href="#ung-ho" class="hidden rounded-sm px-3 py-1.5 text-sm font-medium text-text-muted transition-colors hover:text-text sm:inline-flex">Ủng hộ</a>
        <a href={ctaHref} class="ml-1 inline-flex h-8 items-center gap-1.5 rounded-sm bg-accent px-3.5 text-sm font-medium text-accent-fg transition-[filter] hover:brightness-110">
          Dùng ngay<ArrowRight size={14} />
        </a>
      </nav>
    </div>
  </header>

  <main>
    <!-- ── Hero: copy on one side, the real reader demo on the other ───────────── -->
    <section class="mx-auto max-w-5xl px-4 pt-8 pb-10 sm:px-6 sm:pt-10 sm:pb-12">
      <div class="grid items-center gap-8 lg:grid-cols-2 lg:gap-12">
        <!-- copy -->
        <div class="text-center lg:text-left">
          <!-- the languages themselves, converting → the page's whole job in one line -->
          <div class="flex flex-wrap items-center justify-center gap-1.5 lg:justify-start">
            <span class="rounded-md bg-surface-2 px-2 py-1 text-xs font-medium text-text-muted">日本語</span>
            <span class="rounded-md bg-surface-2 px-2 py-1 text-xs font-medium text-text-muted">한국어</span>
            <span class="rounded-md bg-surface-2 px-2 py-1 text-xs font-medium text-text-muted">中文</span>
            <ArrowRight size={15} class="mx-0.5 text-text-subtle" />
            <span class="rounded-md bg-accent-bg px-2 py-1 text-xs font-semibold text-accent-text">Tiếng Việt</span>
          </div>
          <h1 class="mt-5 text-balance text-[2.5rem] font-bold leading-[1.05] tracking-tight sm:text-5xl">
            Đọc mọi truyện tranh <span class="text-accent">bằng tiếng Việt</span>
          </h1>
          <p class="mx-auto mt-5 max-w-lg text-pretty leading-relaxed text-text-muted lg:mx-0">
            Công cụ đọc &amp; dịch của bạn: mở trang truyện tiếng nước ngoài và đọc ngay bằng tiếng Việt — bản dịch hiện thẳng lên trang, miễn phí, không phải chờ ai.
          </p>
          <div class="mt-7 flex justify-center lg:justify-start">
            <a href={ctaHref} class="inline-flex h-11 w-full items-center justify-center gap-1.5 rounded-sm bg-accent px-6 text-sm font-semibold text-accent-fg transition-[filter] hover:brightness-110 sm:w-auto">
              Dùng ngay, miễn phí<ArrowRight size={16} />
            </a>
          </div>
          <p class="mt-5 text-xs text-text-subtle lg:text-left">Miễn phí · đăng nhập bằng Discord · không cần thẻ.</p>
        </div>

        <!-- product proof: a reader frame, translation overlaid raw → tiếng Việt -->
        <div class="mx-auto w-full max-w-[22rem] lg:mx-0 lg:ml-auto">
          <div class="overflow-hidden rounded-xl bg-surface ring-1 ring-inset ring-white/10">
            <!-- page: aspect locked to the source so the overlay %s line up exactly -->
            <div class="demo-page relative bg-bg" style="aspect-ratio: 1125 / 1600; container-type: inline-size;">
              <img src="/demo/reader-sample.webp" alt="Một trang truyện được dịch sang tiếng Việt" class="absolute inset-0 size-full object-cover" />

              <!-- thin coral scan line just before the overlay appears -->
              <div class="demo-scan pointer-events-none absolute inset-x-0 top-0 z-10 h-px bg-accent"></div>

              <!-- translation overlay — the app's real placement output, fades in (trang gốc → đã dịch) -->
              <div class="demo-overlay absolute inset-0">
                <svg viewBox="0 0 1125 1600" preserveAspectRatio="none" class="absolute inset-0 size-full">
                  <g fill="#fff">
                    {#each maskRects as r (r.join(','))}
                      <rect x={r[0]} y={r[1]} width={r[2]} height={r[3]} rx="2" />
                    {/each}
                  </g>
                </svg>
                {#each overlayLines as t (t.id)}
                  <div class="demo-ov-text" style={`left:${t.left}%;top:${t.top}%;width:${t.w}%;height:${t.h}%;padding:${t.py}cqw ${t.px}cqw`}>
                    <span style={`font-size:${t.fs}cqw;line-height:${t.lh}cqw;-webkit-text-stroke:0.1778cqw #fff`}>{t.text}</span>
                  </div>
                {/each}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ── Platforms — a slim "runs on" strip under the hero, not a full section. ── -->
    <section class="mx-auto max-w-5xl px-4 pb-14 sm:px-6">
      <div class="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-sm text-text-muted">
        <span class="text-text-subtle">Chạy trên</span>
        {#each platforms as p (p.name)}
          <span class="inline-flex items-center gap-1.5 {p.ready ? '' : 'text-text-subtle'}">
            <p.icon size={16} class="shrink-0 text-text-subtle" />{p.name}{#if !p.ready}<span class="ml-1 rounded-full bg-surface-2 px-1.5 py-0.5 text-[10px] font-semibold text-text-subtle">{p.status}</span>{/if}
          </span>
        {/each}
      </div>
    </section>

    <!-- ── Support: core stays free (the legal shield); paying buys speed/status. ── -->
    <section id="ung-ho" class="bg-surface/30">
      <div class="mx-auto max-w-5xl px-4 py-16 sm:px-6">
        <div class="grid items-center gap-10 lg:grid-cols-2 lg:gap-14">
          <!-- the ask — free core stays the headline message -->
          <div>
            <p class="text-xs font-semibold uppercase tracking-[0.14em] text-text-subtle">Ủng hộ</p>
            <h2 class="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">Miễn phí để đọc — ủng hộ nếu muốn dự án đi đường dài</h2>
            <p class="mt-3 text-pretty text-sm leading-relaxed text-text-muted">
              Đọc và dịch cơ bản luôn miễn phí, mãi mãi. Gói Supporter mở chất lượng cao không giới hạn cùng vài đặc quyền, đồng thời giúp trả phí máy chủ — hoàn toàn tự nguyện.
            </p>
            <p class="mt-4 flex items-center gap-1.5 text-xs text-text-subtle">
              <Check size={13} class="shrink-0 text-success-text" />Bỏ qua lúc nào cũng được — bạn vẫn đọc &amp; dịch miễn phí.
            </p>
          </div>

          <!-- one optional membership — a single supporter tier, not a paywall table -->
          <div class="rounded-xl bg-surface p-6 ring-1 ring-inset ring-white/10">
            <div class="flex items-center justify-between">
              <h3 class="text-sm font-semibold">Gói Supporter</h3>
              <span class="rounded-full bg-surface-2 px-2 py-0.5 text-[11px] font-medium text-text-subtle">Tuỳ chọn</span>
            </div>
            <div class="mt-3 flex items-baseline gap-1.5">
              <span class="text-3xl font-bold tracking-tight tabular">39.000đ</span>
              <span class="text-sm text-text-subtle">/tháng</span>
            </div>
            <p class="mt-1 text-xs text-text-subtle">Huỷ bất cứ lúc nào · không ràng buộc</p>
            <ul class="mt-5 space-y-2.5">
              {#each supporterPerks as perk (perk.title)}
                <li class="flex items-start gap-2.5 text-sm text-text-muted">
                  <Check size={16} class="mt-0.5 shrink-0 text-success-text" />
                  <span>{perk.title}</span>
                </li>
              {/each}
            </ul>
            <a href={SUPPORT_HREF} class="mt-6 inline-flex h-11 w-full items-center justify-center gap-1.5 rounded-sm bg-surface-2 px-6 text-sm font-semibold text-text transition-colors hover:bg-hover">
              <Heart size={16} class="text-text-subtle" />Ủng hộ dự án
            </a>
          </div>
        </div>
      </div>
    </section>

    <!-- ── No storage: the legal shield, condensed to one clear statement. ─────── -->
    <section class="mx-auto max-w-5xl px-4 py-14 sm:px-6">
      <div class="mx-auto max-w-2xl text-center">
        <p class="text-xs font-semibold uppercase tracking-[0.14em] text-text-subtle">Riêng tư</p>
        <h2 class="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">Chúng tôi không lưu trữ bất cứ thứ gì</h2>
        <p class="mx-auto mt-3 max-w-lg text-pretty text-sm leading-relaxed text-text-muted">
          Hội Mê Truyện là trình đọc &amp; dịch chạy ngay trên thiết bị của bạn. Mọi nội dung đến từ nguồn bạn tự chọn — chúng tôi không host, không sao chép, không phân phối, và không gửi truyện bạn đọc lên máy chủ nào.
        </p>
      </div>
    </section>

    <!-- ── FAQ ────────────────────────────────────────────────────────────────── -->
    <section class="mx-auto max-w-3xl px-4 py-16 sm:px-6">
      <div class="text-center">
        <p class="text-xs font-semibold uppercase tracking-[0.14em] text-text-subtle">Hỏi đáp</p>
        <h2 class="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">Câu hỏi thường gặp</h2>
      </div>
      <div class="mt-8 space-y-2.5">
        {#each faqs as item (item.q)}
          <details class="group rounded-lg bg-surface/40 ring-1 ring-inset ring-white/5 transition-colors hover:bg-surface/70 open:bg-surface/70">
            <summary class="flex cursor-pointer list-none items-center justify-between gap-4 px-5 py-4 text-[15px] font-medium leading-snug">
              <span>{item.q}</span>
              <ChevronRight size={16} class="shrink-0 text-text-subtle transition-transform group-open:rotate-90" />
            </summary>
            <p class="px-5 pb-5 text-sm leading-relaxed text-text-muted">{item.a}</p>
          </details>
        {/each}
      </div>
    </section>

    <!-- ── Final CTA: coral-tinted band, distinct from the neutral surfaces. ──── -->
    <section class="mx-auto max-w-3xl px-4 pb-16 sm:px-6">
      <div class="flex flex-col items-center gap-4 rounded-xl bg-surface p-8 text-center ring-1 ring-inset ring-white/5 sm:flex-row sm:justify-between sm:text-left">
        <div>
          <h2 class="text-lg font-semibold text-text">Sẵn sàng đọc bằng tiếng Việt?</h2>
          <p class="mt-1 text-sm text-text-muted">Đăng nhập bằng Discord để bắt đầu — chỉ mất vài giây.</p>
        </div>
        <a href={ctaHref} class="inline-flex h-11 shrink-0 items-center justify-center gap-1.5 rounded-sm bg-accent px-7 text-sm font-semibold text-accent-fg transition-[filter] hover:brightness-110">
          Dùng ngay<ArrowRight size={16} />
        </a>
      </div>
    </section>
  </main>

  <footer class="border-t border-border-soft">
    <div class="mx-auto max-w-5xl px-4 py-7 sm:px-6">
      <p class="text-center text-xs leading-relaxed text-text-subtle sm:text-left">
        Hội Mê Truyện là công cụ đọc &amp; dịch truyện cho mục đích cá nhân — không lưu trữ nội dung; truyện đến từ các nguồn bạn tự chọn.
      </p>
      <div class="mt-3 flex items-center justify-center gap-4 text-xs sm:justify-start">
        <span class="text-text-subtle">© {BRAND.name}</span>
        <a href={ctaHref} class="text-text-muted hover:text-text">Đăng nhập</a>
      </div>
    </div>
  </footer>
</div>

<style>
  /* Reader demo. One 7s timeline reveals the translation overlay (trang gốc → đã
     dịch) in sync with a thin coral scan line. The overlay text is the app's own
     SamaritanTall lettering at the exact placement coords it emits. Svelte scopes
     these class + keyframe names together so they stay local. */
  .demo-overlay { opacity: 0; animation: demoReveal 7s ease-in-out infinite; }
  .demo-ov-text {
    position: absolute; display: flex; align-items: center; justify-content: center;
    text-align: center; box-sizing: border-box; overflow: hidden; color: #111; z-index: 1;
  }
  .demo-ov-text span {
    display: block; width: 100%; font-weight: 700; font-family: 'SamaritanTall', serif;
    white-space: pre; paint-order: stroke;
  }

  .demo-scan { opacity: 0; animation: demoScan 7s ease-in-out infinite; }

  @keyframes demoReveal { 0%, 26% { opacity: 0; } 34%, 92% { opacity: 1; } 100% { opacity: 0; } }
  @keyframes demoScan {
    0%, 6% { transform: translateY(0); opacity: 0; }
    10% { opacity: 1; }
    26% { transform: translateY(min(180vw, 620px)); opacity: 1; }
    30%, 100% { opacity: 0; }
  }

  @media (prefers-reduced-motion: reduce) {
    .demo-scan { animation: none; opacity: 0; }
    .demo-overlay { animation: none; opacity: 1; }
  }
</style>
