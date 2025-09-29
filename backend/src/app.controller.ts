import { Controller, Post, Body, Get } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { AppService } from './app.service';

@ApiTags('reviewer')
@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Post('check')
  @ApiOperation({ 
    summary: 'Check review toxicity',
    description: 'Analyze review for toxicity using ML model'
  })
  @ApiResponse({
    status: 200,
    description: 'Toxicity analysis completed successfully',
  })
  @ApiResponse({
    status: 400,
    description: 'Invalid input data',
  })
  @ApiResponse({
    status: 500,
    description: 'Internal server error',
  })
  async checkReview(@Body() body: { text: string }) {
    return this.appService.checkReview(body.text);
  }

  @Post('check-batch')
  @ApiOperation({ 
    summary: 'Check multiple reviews',
    description: 'Analyze array of reviews for toxicity'
  })
  @ApiResponse({
    status: 200,
    description: 'Batch toxicity analysis completed successfully',
  })
  async checkBatchReviews(@Body() body: { texts: string[] }) {
    return this.appService.checkBatchReviews(body.texts);
  }

  @Get('health')
  @ApiOperation({ 
    summary: 'Health check',
    description: 'Check ML model and service availability'
  })
  @ApiResponse({
    status: 200,
    description: 'Service is running normally',
  })
  async healthCheck() {
    return this.appService.healthCheck();
  }
}
