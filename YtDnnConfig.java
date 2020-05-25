package com.opera.search.service.feat;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

import static com.opera.search.service.feat.Feature.FeatBuilder;
import static com.opera.search.service.feat.Feature.FeatBuilder.create;
import static com.opera.search.service.feat.Feature.FeatType.Categorical;
import static com.opera.search.service.feat.Feature.FeatType.ID_MAPPING;

@SuppressWarnings({"WeakerAccess", "ArraysAsListWithZeroOrOneArgument"})
public class YoutubeDNNConfig {
    private static final Logger LOG = LoggerFactory.getLogger(YoutubeDNNConfig.class.getName());

    public static List<Feature> upv1Feats = Arrays.asList(
            FeatBuilder.create("manufacturer").upv1Name("manufacturer").type(Categorical).thd(0.01F).spr(),
            FeatBuilder.create("os").upv1Name("os").type(Categorical).thd(0.01F).spr(),

            FeatBuilder.create("screen_height").upv1Name("screen_height").spr(),
            FeatBuilder.create("screen_width").upv1Name("screen_width").spr(),

            FeatBuilder.create("click_cnt").upv1Name("category_click").spr()
    );

    public static List<Feature> upv3Feats = Arrays.asList();

    public static List<Feature> dpMetaFeats = Arrays.asList(
            FeatBuilder.create("dp_gender").type(Categorical).thd(0.1F * 0.2F).spr()
    );

    public static List<Feature> pubFeats = Arrays.asList();

    public static List<Feature> context = Arrays.asList(
            FeatBuilder.create("cn").type(Categorical).thd(0.01F).spr(),
            FeatBuilder.create("lang").type(Categorical).thd(0.01F).spr(),
            FeatBuilder.create("origin").type(Categorical).spr(),
            FeatBuilder.create("ac").type(Categorical).spr(),
            FeatBuilder.create("app_version").type(Categorical).thd(0.01F).spr(),
            FeatBuilder.create("time_slot").type(Categorical).spr()
    );

    public static List<Feature> impr = Arrays.asList(
            FeatBuilder.create("usr_fl_num").spr(),

            FeatBuilder.create("usr_impr_home").spr(),
            FeatBuilder.create("usr_impr_large").spr(),
            FeatBuilder.create("usr_impr_others").spr(),

            FeatBuilder.create("usr_fr_home").spr(),
            FeatBuilder.create("usr_fr_large").spr(),
            FeatBuilder.create("usr_fr_others").spr()
    );

    /* ---------------------------- Sparse ---------------------------- */

    // featName = spr_$idMapKey

    public static List<Feature> sparseFeatsUser = Arrays.asList(
            // create("spr_device_id").idMapKey("device_id").type(ID_MAPPING).spr(),
            create("spr_fls").idMapKey("pub_id").type(ID_MAPPING).spr(),

            create("spr_news_clks_24h").idMapKey("entry_id_clk").type(ID_MAPPING).spr(),
            create("spr_news_clks_2w").idMapKey("entry_id_clk").type(ID_MAPPING).spr(),
            create("spr_news_clks_others").idMapKey("entry_id_clk").type(ID_MAPPING).spr());

    public static List<Feature> sparseFeatsUserKv = Arrays.asList(
            create("spr_upv1_ct").upv1Name("category_preference").idMapKey("upv1_ct").type(ID_MAPPING).spr(),
            create("spr_upv1_s1_ct").upv1Name("sub_category_preference").idMapKey("upv1_s1_ct").type(ID_MAPPING).spr(),
            create("spr_upv1_s2_ct").upv1Name("sub_sub_category_preference").idMapKey("upv1_s2_ct").type(ID_MAPPING).spr(),
            create("spr_upv1_s3_ct").upv1Name("sub_sub_sub_category_preference").idMapKey("upv1_s3_ct").type(ID_MAPPING).spr(),

            create("spr_upv1_tpc_64").upv1Name("topic_64_preference").idMapKey("upv1_tpc_64").type(ID_MAPPING).spr(),
            create("spr_upv1_tpc_256").upv1Name("topic_256_preference").idMapKey("upv1_tpc_256").type(ID_MAPPING).spr(),
            create("spr_upv1_tpc_2048").upv1Name("topic_2048_preference").idMapKey("upv1_tpc_2048").type(ID_MAPPING).spr(),

            create("spr_upv1_tl_kw").upv1Name("title_keyword_preference").idMapKey("upv1_tl_kw").type(ID_MAPPING).spr(),
            create("spr_upv1_kw").upv1Name("keyword_preference").idMapKey("upv1_kw").type(ID_MAPPING).spr(),
            create("spr_upv1_sp_kw").upv1Name("supervised_keyword_preference").idMapKey("upv1_sp_kw").type(ID_MAPPING).spr(),

            create("spr_upv3_tpc").upv3Name("nl_topic").idMapKey("upv3_tpc").type(ID_MAPPING).spr(),
            create("spr_upv3_ent_v2").upv3Name("nl_key_entities_v2").idMapKey("upv3_ent_v2").type(ID_MAPPING).spr(),
            create("spr_upv3_dm").upv3Name("nl_domain").idMapKey("upv3_dm").type(ID_MAPPING).spr(),

            create("spr_upv3_st_tpc").upv3Name("ns_topic").idMapKey("upv3_st_tpc").type(ID_MAPPING).spr(),
            create("spr_upv3_st_ent_v2").upv3Name("ns_key_entities_v2").idMapKey("upv3_st_ent_v2").type(ID_MAPPING).spr(),
            create("spr_upv3_st_dm").upv3Name("ns_domain").idMapKey("upv3_st_dm").type(ID_MAPPING).spr()
    );

    public static List<List<String>> sameEmbFeatNameLists = Arrays.asList(Arrays.asList(
            "spr_news_clks_24h",
            "spr_news_clks_2w",
            "spr_news_clks_others"
    ));

    public static List<Feature> sparseFeatsPub = Arrays.asList();
    public static List<Feature> sparseFeatsPubKv = Arrays.asList();

    public static List<Feature> featsForOffline = new ArrayList<>();
    public static Set<String> featNamesForGbdt;
    public static Set<String> featNamesForSparse;

    public static Set<String> featNamesSparseKv = new HashSet<>();

    static {

        featsForOffline.addAll(upv1Feats);
        featsForOffline.addAll(upv3Feats);

        featsForOffline.addAll(dpMetaFeats);

        featsForOffline.addAll(pubFeats);

        featsForOffline.addAll(context);

        featsForOffline.addAll(impr);

        featsForOffline.addAll(sparseFeatsUser);
        featsForOffline.addAll(sparseFeatsUserKv);
        featsForOffline.addAll(sparseFeatsPub);
        featsForOffline.addAll(sparseFeatsPubKv);

        //

        featNamesSparseKv.addAll(sparseFeatsUserKv.stream().map(f -> f.name).collect(Collectors.toList()));
        featNamesSparseKv.addAll(sparseFeatsPubKv.stream().map(f -> f.name).collect(Collectors.toList()));

        /////////

        featNamesForGbdt = featsForOffline.stream().filter(f -> f.useForGbdt).map(f -> f.name).collect(Collectors.toSet());
        featNamesForSparse = featsForOffline.stream().filter(f -> f.useForSparse).map(f -> f.name).collect(Collectors.toSet());

        LOG.info("YoutubeDNNConfig, offline features, size: {}, feats: {}", featsForOffline.size(),
                String.join("\n", featsForOffline.stream().map(Feature::toString).collect(Collectors.toList())));
        LOG.info("YoutubeDNNConfig, offline sparse features, size: {}, feats: {}\n", featNamesForSparse.size(),
                String.join("\n", featNamesForSparse));
    }
}
