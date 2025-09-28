<template>
  <div class="product-features">
    <h3>Характеристики</h3>
    <ul class="feature-list">
      <li 
        v-for="feature in features" 
        :key="feature.name"
        class="feature-item"
      >
        <span class="feature-label">{{ feature.name }}:</span>
        <span class="feature-value">{{ feature.value }}</span>
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'ProductFeatures',
  props: {
    features: {
      type: Array,
      required: true,
      default: () => []
    }
  }
}
</script>

<style scoped>
.product-features {
  margin-bottom: 32px;
}

.product-features h3 {
  font-size: 20px;
  font-weight: 600;
  color: var(--wb-dark);
  margin-bottom: 16px;
}

.feature-list {
  list-style: none;
}

.feature-item {
  display: flex;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid var(--wb-gray-200);
}

.feature-item:last-child {
  border-bottom: none;
}

.feature-label {
  font-weight: 600;
  color: var(--wb-gray-600);
  min-width: 140px;
  flex-shrink: 0;
}

.feature-value {
  color: var(--wb-dark);
  flex: 1;
}

@media (max-width: 768px) {
  .feature-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }
  
  .feature-label {
    min-width: auto;
    font-size: 14px;
  }
  
  .feature-value {
    font-size: 14px;
  }
}
</style>
