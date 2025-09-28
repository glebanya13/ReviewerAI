<template>
  <div class="reviews-page">
    <ReviewsHeader />
    
    <ProductInfo :product="product" />

    <div class="reviews-list-section">
      <div class="container">
        <ReviewsFilters @sort-changed="handleSortChange" />
        <ReviewsList :reviews="sortedReviews" />
      </div>
    </div>
  </div>
</template>

<script>
import ReviewsHeader from './components/ReviewsHeader.vue'
import ProductInfo from './components/ProductInfo.vue'
import ReviewsFilters from './components/ReviewsFilters.vue'
import ReviewsList from './components/ReviewsList.vue'

export default {
  name: 'ReviewsPage',
  components: {
    ReviewsHeader,
    ProductInfo,
    ReviewsFilters,
    ReviewsList
  },
  data() {
    return {
      sortBy: 'newest',
      product: {
        id: 35433019,
        title: 'Комбинезон утепленный с начесом для новорожденных Супер пупс',
        rating: 4.5,
        reviewsCount: 127,
        images: [
          '/images/preview.webp',
          '/images/1.webp',
          '/images/1.webp',
          '/images/1.webp',
          '/images/1.webp'
        ]
      },
      reviews: [
        {
          id: 1,
          name: 'Анна',
          avatar: '/images/1.webp',
          rating: 5,
          date: '2024-01-15',
          text: 'Отличный комбинезон! Очень теплый и качественный. Ребенку комфортно, не потеет. Рекомендую!',
          photos: ['/images/1.webp', '/images/1.webp']
        },
        {
          id: 2,
          name: 'Мария',
          avatar: '/images/1.webp',
          rating: 4,
          date: '2024-01-10',
          text: 'Хороший комбинезон, но размер маловат. Качество отличное, материал приятный.',
          photos: []
        },
        {
          id: 3,
          name: 'Елена',
          avatar: '/images/1.webp',
          rating: 5,
          date: '2024-01-08',
          text: 'Супер! Очень довольна покупкой. Ребенок носит с удовольствием, не снимает.',
          photos: ['/images/1.webp'],
          response: {
            author: 'Команда Швейной Фабрики "Супер Пупс"',
            date: '2024-01-09',
            text: 'Спасибо за отзыв! Рады, что товар вам понравился!'
          }
        },
        {
          id: 4,
          name: 'Ольга',
          avatar: '/images/1.webp',
          rating: 3,
          date: '2024-01-05',
          text: 'Нормальный комбинезон, но застежки не очень удобные. В остальном все хорошо.',
          photos: []
        },
        {
          id: 5,
          name: 'Ирина',
          avatar: '/images/1.webp',
          rating: 5,
          date: '2024-01-03',
          text: 'Превосходное качество! Очень довольна. Ребенок в нем не мерзнет на прогулках.',
          photos: ['/images/1.webp', '/images/1.webp']
        }
      ]
    }
  },
  computed: {
    sortedReviews() {
      const sorted = [...this.reviews];
      switch (this.sortBy) {
        case 'newest':
          return sorted.sort((a, b) => new Date(b.date) - new Date(a.date));
        case 'oldest':
          return sorted.sort((a, b) => new Date(a.date) - new Date(b.date));
        case 'rating-high':
          return sorted.sort((a, b) => b.rating - a.rating);
        case 'rating-low':
          return sorted.sort((a, b) => a.rating - b.rating);
        default:
          return sorted;
      }
    }
  },
  methods: {
    handleSortChange(sortBy) {
      this.sortBy = sortBy;
    }
  }
}
</script>

<style scoped>
.reviews-page {
  min-height: 100vh;
  background-color: var(--mo-neutral-970);
}

.reviews-list-section {
  padding: 24px 0;
}

@media (max-width: 480px) {
  .container {
    padding: 0 12px;
  }
  
  .reviews-list-section {
    padding: 16px 0;
  }
}
</style>
