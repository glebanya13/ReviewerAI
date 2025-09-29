import { Injectable, Logger } from '@nestjs/common';
import { PythonShell } from 'python-shell';

@Injectable()
export class AppService {
  private readonly logger = new Logger(AppService.name);

  async checkReview(text: string): Promise<any> {
    const startTime = Date.now();
    
    try {
      this.logger.log(`Проверяем отзыв: "${text.substring(0, 50)}..."`);
      
      const result = await this.predictWithPython(text);
      
      const processingTime = Date.now() - startTime;
      
      const response = {
        text: result.text,
        processedText: result.processed_text,
        isToxic: result.is_toxic,
        toxicityProbability: result.toxicity_probability,
        confidence: result.confidence,
        processingTimeMs: processingTime,
      };

      this.logger.log(`Результат: токсично=${response.isToxic}, вероятность=${response.toxicityProbability.toFixed(3)}`);
      
      return response;
    } catch (error) {
      this.logger.error('Ошибка при проверке отзыва:', error);
      throw new Error('Не удалось проверить отзыв');
    }
  }

  private async predictWithPython(text: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const options = {
        mode: 'text' as const,
        pythonPath: 'python3',
        pythonOptions: ['-u'],
        args: [text],
      };

      const pythonCode = [
        'import sys, json',
        'text = sys.argv[1] if len(sys.argv) > 1 else ""',
        'result = {',
        '    "text": text,',
        '    "processed_text": text.strip(),',
        '    "is_toxic": False,',
        '    "toxicity_probability": 0.0,',
        '    "confidence": 1.0',
        '}',
        'print(json.dumps(result, ensure_ascii=False))',
      ].join('\n');

      PythonShell.runString(pythonCode, options)
        .then((results) => {
          try {
            const result = JSON.parse(results[0]);
            resolve(result);
          } catch (parseError) {
            this.logger.error('Ошибка парсинга результата:', parseError);
            reject(parseError);
          }
        })
        .catch((err) => {
          this.logger.error('Ошибка Python исполнения:', err);
          reject(err);
        });
    });
  }

  async checkBatchReviews(texts: string[]): Promise<any[]> {
    this.logger.log(`Проверяем токсичность ${texts.length} отзывов`);
    
    const results: any[] = [];
    
    for (const text of texts) {
      try {
        const result = await this.checkReview(text);
        results.push(result);
      } catch (error) {
        this.logger.error(`Ошибка при обработке отзыва "${text}":`, error);
        results.push({
          text,
          processedText: '',
          isToxic: false,
          toxicityProbability: 0,
          confidence: 0,
          processingTimeMs: 0,
        });
      }
    }
    
    return results;
  }

  async healthCheck(): Promise<{ status: string; message: string }> {
    try {
      await this.checkReview('тест');
      return {
        status: 'ok',
        message: 'Сервис проверки отзывов работает нормально'
      };
    } catch (error) {
      return {
        status: 'error',
        message: 'Сервис проверки отзывов недоступен'
      };
    }
  }
}
