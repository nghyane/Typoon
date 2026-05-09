import {DiscordSDK, DiscordSDKMock, type IDiscordSDK} from '@discord/embedded-app-sdk'

const CLIENT_ID = import.meta.env.VITE_DISCORD_CLIENT_ID as string

export const isDiscordActivity =
  window.location.hostname.endsWith('.discordsays.com')

let discordSdk: IDiscordSDK

if (isDiscordActivity) {
  discordSdk = new DiscordSDK(CLIENT_ID)
} else {
  discordSdk = new DiscordSDKMock(CLIENT_ID, null, null, null)
}

export {discordSdk}
