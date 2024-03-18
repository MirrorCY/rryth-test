import { Context, Schema } from 'koishi'
import { } from "@koishijs/censor"


export const name = 'rryth-test'

export function apply(ctx: Context, config: Config) {
  // write your plugin here
  const resolution = (resolution: string): number => +resolution & ~63
  ctx
    .command('rryth-test <prompt:text>', '人人有图画测试服 v0.0.1')
    .usage('这里会不定期更新一些有意思的功能，也随时会关闭服务\n这次的是 LCM，它跑的真的很快')
    .example('rt -b 1 -x 2333 1 girl')
    .alias('rt')
    .option('seed', '-x <seed:number> 种子')
    .option('cfg_scale', '-c <cfg_scale:number> 无分类器引导 0-10 默认 10', { fallback: 10 })
    .option('width', '-w <width:number> 宽', { fallback: 768, type: resolution })
    .option('height', '-g <height:number> 高', { fallback: 512, type: resolution })
    .option('batch', '-b <batch:number> 批量', { fallback: config.batch })

    .action(async ({ session, options }, prompt) => {
      if (!prompt) return session.execute('help rryth-test')
      const request: Request = { prompt, ...options }
      const rr = async (request: Request) => {
        const { images } = await ctx.http.post('http://api.rryth.com:42420', request, { headers: { 'api': 'LCM' } })
          .catch(e => console.log(e)) as Response
        return images.map(image => {
          return config.censor
            ? <censor><img src={'data:image/png;base64,' + image}></img></censor>
            : <img src={'data:image/png;base64,' + image}></img>
        })
      }
      const images = await rr(request)
      session.send(images)
    })
}

export interface Config {
  censor?: boolean
  batch?: number
}

export const Config: Schema<Config> = Schema.object({
  censor: Schema.boolean().description('是否启用图像审查。').default(true),
  batch: Schema.number().description('默认出图数量。').default(4).max(4).min(1).role('slider'),
})

interface Request {
  prompt: string
  seed?: number
  np?: string
  cfg_scale?: number
  width?: number
  height?: number
  steps?: number
  batch?: number
}

interface Response {
  images: string[] // base64 encoded images
}