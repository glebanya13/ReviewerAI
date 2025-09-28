<template>
  <div class="product-price">
    <span class="current-price">{{ formatPrice(price) }} ₽</span>
    <span v-if="oldPrice" class="old-price">{{ formatPrice(oldPrice) }} ₽</span>
    <span v-if="discount" class="discount">-{{ discount }}%</span>
  </div>
</template>

<script>
export default {
  name: 'ProductPrice',
  props: {
    price: {
      type: Number,
      required: true
    },
    oldPrice: {
      type: Number,
      default: null
    },
    discount: {
      type: Number,
      default: null
    }
  },
  methods: {
    formatPrice(price) {
      return new Intl.NumberFormat('ru-RU').format(price)
    }
  }
}
</script>

<style scoped>
.product-price {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 24px;
}

.current-price {
  font-size: 32px;
  font-weight: 700;
  color: var(--mo-button-primary-bg-default);
}

.old-price {
  font-size: 20px;
  color: var(--mo-text-icon-secondary-default);
  text-decoration: line-through;
}

.discount {
  background: #ff3333;
  color: var(--mo-neutral-1000);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 600;
}

@media (max-width: 768px) {
  .current-price {
    font-size: 28px;
  }
  
  .old-price {
    font-size: 18px;
  }
}
</style>
