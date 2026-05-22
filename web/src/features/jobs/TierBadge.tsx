// Tier display chips — match the colors of the Free/Supporter/Pro/Unlimited
// ladder for instant recognition in the sidebar / settings.

import { Crown, Sparkles, Zap, Heart } from 'lucide-react'
import { Tag, type TagTone } from '@shared/ui/primitives'
import type { ApiTierInfo } from '@shared/api/api'

const TIER_TONE: Record<string, TagTone> = {
  free:      'neutral',
  supporter: 'info',
  pro:       'accent',
  unlimited: 'success',
}

const TIER_ICON: Record<string, React.ComponentType<{ size?: number }>> = {
  free:      Sparkles,
  supporter: Heart,
  pro:       Zap,
  unlimited: Crown,
}

interface Props {
  tier:       Pick<ApiTierInfo, 'id' | 'name'>
  className?: string
}

export function TierBadge({ tier, className }: Props) {
  const Icon = TIER_ICON[tier.id] ?? Sparkles
  return (
    <Tag tone={TIER_TONE[tier.id] ?? 'neutral'} className={className}>
      <Icon size={11} />
      {tier.name}
    </Tag>
  )
}
