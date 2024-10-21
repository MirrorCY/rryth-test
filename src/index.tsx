import { Context, Schema } from 'koishi'
import { } from "@koishijs/censor"


export const name = 'rryth-test'

export function apply(ctx: Context, config: Config) {
  // write your plugin here
  const resolution = (resolution: string): number => +resolution & ~63
  ctx
    .command('rryth-test <prompt:text>', '人人有图画测试服 v0.0.7')
    .usage('这里会不定期更新一些有意思的功能，也随时会关闭服务\n这次的是 FLUX，它好像啥风格都能画。rt 为原版、rtx 为竖版、rta 为动漫风格。')
    .example('rta -x 2333 1girl')
    .alias('rt', { options: { width: 768, height: 512 } })
    .alias('rtx', { options: { width: 512, height: 768 } })
    .alias('rta', { options: { width: 512, height: 768 }, args: ['anime'] })
    .option('seed', '-x <seed:number> 种子')
    // .option('cfg_scale', '-c <cfg_scale:number> 无分类器引导 0-10 默认 10', { fallback: 1 })
    .option('width', '-w <width:number> 宽', { fallback: 768, type: resolution })
    .option('height', '-g <height:number> 高', { fallback: 512, type: resolution })
    // .option('batch', '-b <batch:number> 批量', { fallback: config.batch })
    // .option('iterations', '-i <iterations:number> 多来几下', { authority: 2 })
    .action(async ({ session, options }, ...prompts) => {
      const prompt = prompts.join(', ')
      if (!prompt) return session.execute('help rt')
      const request: Request = { prompt, ...options }
      const rr = async (request: Request) => {
        const { images } = await ctx.http.post('https://rr.elchapo.cn', request, { headers: { 'api': 'FLUX.0' } })
          .catch(e => ctx.logger.error(e)) as Response
        return images.map(image => {
          return config.censor
            ? <censor><img src={'data:image/png;base64,' + image}></img></censor>
            : <img src={'data:image/png;base64,' + image}></img>
        })
      }
      // options.iterations = options.iterations > 10 ? 10 : options.iterations || 1
      // for (let i = 0; i < options.iterations; i++) {
      //   const images = await rr(request)
      //   session.send(images)
      // }
      const images = await rr(request)
      session.send(images)
    })
}

export interface Config {
  censor?: boolean
  batch?: number
}

export const Config: Schema<Config> = Schema.object({
  censor: Schema.boolean().description('是否启用图像审查。').default(false),
  batch: Schema.number().description('默认出图数量。').default(1).max(4).min(1).role('slider').disabled().hidden(),
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
  images: string[]
}
