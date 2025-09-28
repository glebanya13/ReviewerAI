<template>
  <div class="product-page">
    <div class="product-container">
      <div class="product-gallery">
        <ProductGallery :images="product.images" />
      </div>

      <div class="product-info">
        <h1 class="product-title">{{ product.title }}</h1>

        <ProductPrice :price="product.price" :old-price="product.oldPrice" :discount="product.discount" />

        <ProductActions :product-id="product.id" @cart-added="handleCartAdded" @favorite-added="handleFavoriteAdded"
          @favorite-removed="handleFavoriteRemoved" />

        <ProductRating :rating="product.rating" :reviews-count="product.reviewsCount" @go-to-reviews="goToReviews" />

        <ProductFeatures :features="product.features" />

        <div class="product-description">
          <h3>Описание</h3>
          <p>{{ product.description }}</p>
        </div>

        <ProductReviews :overall-rating="product.rating" :total-reviews="product.reviewsCount" />
      </div>
    </div>
  </div>
</template>

<script>
import ProductGallery from './components/ProductGallery.vue'
import ProductPrice from './components/ProductPrice.vue'
import ProductActions from './components/ProductActions.vue'
import ProductFeatures from './components/ProductFeatures.vue'
import ProductRating from './components/ProductRating.vue'
import ProductReviews from './components/ProductReviews.vue'

export default {
  name: 'ProductPage',
  components: {
    ProductGallery,
    ProductPrice,
    ProductActions,
    ProductFeatures,
    ProductRating,
    ProductReviews
  },
  data() {
    return {
      product: {
        id: 35433019,
        title: 'Комбинезон утепленный с начесом для новорожденных Супер пупс',
        price: 1253,
        oldPrice: null,
        discount: null,
        rating: 4.5,
        reviewsCount: 127,
        images: [
          '/images/preview.webp',
          '/images/1.webp',
          '/images/1.webp',
          '/images/1.webp',
          '/images/1.webp'
        ],
        features: [
          { name: 'Бренд', value: 'Супер пупс' },
          { name: 'Пол', value: 'Унисекс' },
          { name: 'Возраст', value: '0-6 месяцев' },
          { name: 'Материал', value: '100% хлопок' },
          { name: 'Сезон', value: 'Зима' },
          { name: 'Утеплитель', value: 'Начес' },
          { name: 'Застежка', value: 'Кнопки' },
          { name: 'Стирка', value: 'Машинная стирка' }
        ],
        description: 'Утепленный комбинезон для новорожденных с начесом. Мягкий и теплый материал обеспечивает комфорт малышу. Удобная застежка на кнопки для легкой смены подгузника. Подходит для прогулок в холодную погоду.'
      }
    }
  },
  methods: {
    handleCartAdded(productId) {
    },
    handleFavoriteAdded(productId) {
    },
    handleFavoriteRemoved(productId) {
    },
    goToReviews() {
      this.$router.push({ name: 'Reviews' })
    }
  }
}
</script>

<style scoped>
.product-page {
  background: var(--mo-neutral-970);
  min-height: 100vh;
  padding: 16px 0;
}

.product-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 16px;
}

.product-gallery {
  background: var(--mo-neutral-1000);
  border-radius: 12px;
  padding: 16px;
  height: fit-content;
}

.product-info {
  background: var(--mo-neutral-1000);
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 12px var(--mo-neutral-0A10);
  border: 1px solid var(--mo-neutral-900);
}

.product-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--mo-text-icon-primary-default);
  margin-bottom: 12px;
  line-height: 1.3;
}

.product-description {
  margin-top: 32px;
}

.product-description h3 {
  font-size: 20px;
  font-weight: 600;
  color: var(--mo-text-icon-primary-default);
  margin-bottom: 12px;
}

.product-description p {
  color: var(--mo-text-icon-secondary-default);
  line-height: 1.6;
}

@media (max-width: 768px) {
  .product-container {
    grid-template-columns: 1fr;
    gap: 16px;
    padding: 0 12px;
  }

  .product-title {
    font-size: 16px;
    line-height: 1.4;
  }
}
</style>
