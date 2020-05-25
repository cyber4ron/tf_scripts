package com.opera.search.service.feat;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

import static com.opera.search.service.feat.Config.crossFeat;
import static com.opera.search.service.feat.Config.crossFeatUpv3;
import static com.opera.search.service.feat.Feature.FeatBuilder;
import static com.opera.search.service.feat.Feature.FeatBuilder.create;
import static com.opera.search.service.feat.Feature.FeatType.Categorical;
import static com.opera.search.service.feat.Feature.FeatType.ID_MAPPING;
import static com.opera.search.service.feat.Feature.copy;
import static com.opera.search.service.feat.Stats.REC_FEAT_TOP_NUM;

/**
 * Feature Candidates
 */
@SuppressWarnings("WeakerAccess")
public class DeepFMConfig {
    private static final Logger LOG = LoggerFactory.getLogger(DeepFMConfig.class.getName());

    /* ---------------------------------------------------- Match ---------------------------------------------------- */

    public static List<Feature> prefKVFeats = Arrays.asList(

            /* ------------------------ upv1 ------------------------ */
            create("upv1_tpc_2048").newsName("topic2048").upv1Name("topic_2048_preference"),
            create("upv1_s1_ct").newsName("sub_category").upv1Name("sub_category_preference"),
            create("upv1_kw").newsName("keywords").upv1Name("keyword_preference"),

            /* ------------------------ upv3 ------------------------ */
            create("upv3_dm").newsName("domain").upv3Name("nl_domain")
    );

    public static List<Feature> prefKvFeatsMean = prefKVFeats.stream().map(f -> copy(f).name(String.format("%s_mean", f.name))).collect(Collectors.toList());
    public static List<Feature> prefKvFeatsMax = prefKVFeats.stream().map(f -> copy(f).name(String.format("%s_max", f.name))).collect(Collectors.toList());

    /* ---------------------------------------------------- Upv1/Upv3/DpMeta ---------------------------------------------------- */

    public static List<Feature> upv1Feats = Arrays.asList(
            create("manufacturer").upv1Name("manufacturer").type(Categorical).thd(0.01F).spr(),
            create("os").upv1Name("os").type(Categorical).thd(0.01F).spr(),

            create("screen_height").upv1Name("screen_height"),
            create("screen_width").upv1Name("screen_width"),

            create("click_cnt").upv1Name("category_click")
    );

    public static List<Feature> upv3Feats = Arrays.asList();

    public static List<Feature> dpMetaFeats = Arrays.asList(
            create("dp_gender").type(Categorical).thd(0.1F * 0.2F).spr(),
            create("dp_entry_id").type(Categorical).thd(0.1F * 0.05F).spr()
    );

    /* ---------------------------------------------------- Pub/Recaller/Ctx/Impr ---------------------------------------------------- */

    public static List<Feature> pubFeats = Arrays.asList(
            // publisher history stats
            create("impr"),
            create("clk"),
            create("fl"),
            create("unfl"),
            create("fl_rate"),
            create("hot_score"),

            // publisher detail
            create("group_id").type(Categorical).thd(0.01F).spr(),
            create("real_subs"),
            create("real_posts"),
            create("desc_len"),

            // publisher detail additional
            create("pub_id_type").type(Categorical).thd(0.01F).spr(),
            create("is_no_logo").type(Categorical).thd(0.01F).spr(),
            create("is_no_post").type(Categorical).thd(0.01F).spr(),

            // publisher profile
            create("pub_prf_spt"),
            create("pub_prf_egn")
    );

    public static List<Feature> context = Arrays.asList(
            create("cn").type(Categorical).thd(0.01F).spr(),
            create("origin").type(Categorical).spr(),
            create("ac").type(Categorical).spr(),
            create("app_version").type(Categorical).thd(0.01F).spr(),
            create("time_slot").type(Categorical).spr(),
            create("pos_of_card").type(Categorical).spr()
    );

    public static List<Feature> impr = Arrays.asList(
            create("impr_home"),
            create("impr_large"),
            create("impr_others"),

            create("impr_ts_home"),
            create("impr_ts_large"),
            create("impr_ts_others"),

            create("usr_fl_num"),

            create("usr_impr_home"),
            create("usr_impr_large"),
            create("usr_impr_others"),

            create("usr_fr_home"),
            create("usr_fr_large"),
            create("usr_fr_others")
    );

    /* ---------------------------------------------------- Stats ---------------------------------------------------- */

    // cross feats for recaller
    public static List<Feature> statsFeatsRec = new ArrayList<>();

    public static String[] objs = new String[]{"pub_id", "group_id"};
    public static String[] objs_pub_id = new String[]{"pub_id"};
    public static String[] aggs = new String[]{"fr", "fl"};

    static {
        for (String obj : objs) {
            for (String agg : aggs) {
                for (int i = 1; i <= REC_FEAT_TOP_NUM; i++) {
                    statsFeatsRec.add(FeatBuilder.create(String.format("rec_src_v2_%s_%s_%d", obj, agg, i)));
                    statsFeatsRec.add(FeatBuilder.create(String.format("rec_src_v2_all_%s_%s_%d", obj, agg, i)));
                    statsFeatsRec.add(FeatBuilder.create(String.format("rec_src_rank_v2_all_%s_%s_%d", obj, agg, i)));
                }
            }
        }
    }

    public static List<Feature> statsFeatsImpr = new ArrayList<>();

    static {
        for (String agg : aggs) {
            for (String origin : new String[]{"home", "large", "others"}) {
                statsFeatsImpr.add(FeatBuilder.create(String.format("org_impr_%s_pub_id_%s", origin, agg)));
                statsFeatsImpr.add(FeatBuilder.create(String.format("org_impr_%s_all_pub_id_%s", origin, agg)));
            }
        }
    }

    public static List<Feature> statsFeatsUpV2 = new ArrayList<>();

    static {
        // upv1
        statsFeatsUpV2.addAll(crossFeat("category", "category_preference", "upv1_ct", objs_pub_id, aggs, 3));
        statsFeatsUpV2.addAll(crossFeat("sub_category", "sub_category_preference", "upv1_s1_ct", objs_pub_id, aggs, 3));

        statsFeatsUpV2.addAll(crossFeat("topic2048", "topic_2048_preference", "upv1_tpc_2048", objs_pub_id, aggs, 3));

        statsFeatsUpV2.addAll(crossFeat("upv1_tl_kw", "title_keyword_preference", "upv1_tl_kw", objs_pub_id, aggs, 3));
        statsFeatsUpV2.addAll(crossFeat("upv1_sp_kw", "supervised_keyword_preference", "upv1_sp_kw", objs_pub_id, aggs, 3));

        // upv3
        statsFeatsUpV2.addAll(crossFeatUpv3("upv3_ent_v2", "nl_key_entities_v2", "upv3_ent_v2", objs_pub_id, aggs, 3));

        statsFeatsUpV2.addAll(crossFeatUpv3("upv3_st_tpc", "ns_topic", "upv3_st_tpc", objs_pub_id, aggs, 3));
        statsFeatsUpV2.addAll(crossFeatUpv3("upv3_st_ent_v2", "ns_key_entities_v2", "upv3_st_ent_v2", objs_pub_id, aggs, 3));

        // upv1 meta
        statsFeatsUpV2.addAll(crossFeat("manufacturer", "manufacturer", "mft", objs, aggs, 1));
        statsFeatsUpV2.addAll(crossFeat("os", "os", "os", objs, aggs, 1));
        statsFeatsUpV2.addAll(crossFeat("screen_height", "screen_height", "scr_hgt", objs, aggs, 1));
        statsFeatsUpV2.addAll(crossFeat("click_cnt", "", "clk_cnt", objs, aggs, 1));
    }

    public static List<Feature> statsFeatsCtx = new ArrayList<>();

    static {
        statsFeatsCtx.addAll(crossFeat("cn", "", "cn", objs, aggs, 1));
        statsFeatsCtx.addAll(crossFeat("ac", "", "ac", objs, aggs, 1));
        statsFeatsCtx.addAll(crossFeat("origin", "", "origin", objs, aggs, 1));
        statsFeatsCtx.addAll(crossFeat("app_version", "", "app_ver", objs, aggs, 1));
    }

    /* ---------------------------------------------------- Sparse ---------------------------------------------------- */

    public static List<Feature> sparseFeatsUser = Arrays.asList(
            // create("spr_device_id").idMapKey("device_id").type(ID_MAPPING).spr(),
            create("spr_fls").idMapKey("pub_id").type(ID_MAPPING).spr(),
            create("spr_fls_pub_grp").idMapKey("pub_grp_id").type(ID_MAPPING).spr(),

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

    public static List<Feature> sparseFeatsPub = Arrays.asList(
            create("spr_pub_id").idMapKey("pub_id").type(ID_MAPPING).spr(),
            create("spr_pub_grp_id").idMapKey("pub_grp_id").type(ID_MAPPING).spr());

    public static List<Feature> sparseFeatsPubKv = Arrays.asList(
            create("spr_pub_tpc").newsName("topic").idMapKey("pub_tpc").type(ID_MAPPING).spr(),
            create("spr_pub_tpc_64").newsName("topic64").idMapKey("pub_tpc_64").type(ID_MAPPING).spr(),
            create("spr_pub_tpc_256").newsName("topic256").idMapKey("pub_tpc_256").type(ID_MAPPING).spr(),
            create("spr_pub_tpc_2048").newsName("topic2048").idMapKey("pub_tpc_2048").type(ID_MAPPING).spr(),

            create("spr_pub_ct").newsName("category").idMapKey("pub_ct").type(ID_MAPPING).spr(),
            create("spr_pub_s1_ct").newsName("sub_category").idMapKey("pub_s1_ct").type(ID_MAPPING).spr(),
            create("spr_pub_s2_ct").newsName("sub_sub_category").idMapKey("pub_s2_ct").type(ID_MAPPING).spr(),
            create("spr_pub_s3_ct").newsName("sub_sub_sub_category").idMapKey("pub_s3_ct").type(ID_MAPPING).spr(),

            create("spr_pub_ent").newsName("key_entities").idMapKey("pub_ent").type(ID_MAPPING).spr(),
            create("spr_pub_sp_kw_v2").newsName("supervised_keywords_v2").idMapKey("pub_sp_kw_v2").type(ID_MAPPING).spr(),
            create("spr_pub_tpc_v2").newsName("topic_v2").idMapKey("pub_tpc_v2").type(ID_MAPPING).spr(),

            create("spr_pub_tl_kw").newsName("title_keywords").idMapKey("pub_tl_kw").type(ID_MAPPING).spr(),
            create("spr_pub_kw").newsName("keywords").idMapKey("pub_kw").type(ID_MAPPING).spr(),
            create("spr_pub_sp_kw").newsName("supervised_keywords").idMapKey("pub_sp_kw").type(ID_MAPPING).spr(),

            create("spr_pub_dm").newsName("domain").idMapKey("pub_dm").type(ID_MAPPING).spr()
    );

    /* -------------------------------------------------------------------------------------------------------- */

    // todo
    public static List<Feature> sparseFeatsPubRecallerKv = Arrays.asList(
            create("spr_rec_src_rank").idMapKey("rec_src_rank").type(ID_MAPPING).spr()
    );

    public static List<Feature> featsForOffline = new ArrayList<>();

    public static Set<String> featNamesSparse;
    public static Set<String> featNamesSparseKv = new HashSet<>(); // value是sorted linked hash map

    public static List<List<String>> sameEmbFeatNameLists = Arrays.asList(Arrays.asList(
            "spr_news_clks_24h",
            "spr_news_clks_2w",
            "spr_news_clks_others"
    ), Arrays.asList(
            "spr_fls",
            "spr_pub_id"
    ), Arrays.asList(
            "spr_fls_pub_grp",
            "spr_pub_grp_id"
    ));

    static {
        featsForOffline.addAll(prefKVFeats);
        featsForOffline.addAll(prefKvFeatsMean);
        featsForOffline.addAll(prefKvFeatsMax);

        featsForOffline.addAll(upv1Feats);
        featsForOffline.addAll(upv3Feats);

        featsForOffline.addAll(dpMetaFeats);

        featsForOffline.addAll(pubFeats);

        featsForOffline.addAll(context);

        featsForOffline.addAll(statsFeatsRec);
        featsForOffline.addAll(statsFeatsCtx);

        featsForOffline.addAll(impr);

        featsForOffline.addAll(statsFeatsUpV2);
        featsForOffline.addAll(statsFeatsImpr);

        featsForOffline.addAll(sparseFeatsUser);
        featsForOffline.addAll(sparseFeatsUserKv);

        featsForOffline.addAll(sparseFeatsPub);
        featsForOffline.addAll(sparseFeatsPubKv);

        featsForOffline.addAll(sparseFeatsPubRecallerKv);

        //

        featNamesSparse = featsForOffline.stream().filter(f -> f.useForSparse).map(f -> f.name).collect(Collectors.toSet());

        featNamesSparseKv.addAll(sparseFeatsUserKv.stream().map(f -> f.name).collect(Collectors.toList()));
        featNamesSparseKv.addAll(sparseFeatsPubKv.stream().map(f -> f.name).collect(Collectors.toList()));
        featNamesSparseKv.addAll(sparseFeatsPubRecallerKv.stream().map(f -> f.name).collect(Collectors.toList()));

        //

        LOG.info("DeepFMConfig, feat list size: {}, feat name set size: {}", featsForOffline.size(),
                featsForOffline.stream().map(f -> f.name).collect(Collectors.toSet()).size());

        LOG.info("DeepFMConfig, offline features, size: {}, feats: {}", featsForOffline.size(),
                String.join("\n", featsForOffline.stream().map(Feature::toString).collect(Collectors.toList())));

        LOG.info("DeepFMConfig, offline sparse features, size: {}, feats: {}\n", featNamesSparse.size(),
                String.join("\n", featNamesSparse));

        LOG.info("DeepFMConfig, offline sparse kv features, size: {}, feats: {}\n", featNamesSparseKv.size(),
                String.join("\n", featNamesSparseKv));
    }
}
